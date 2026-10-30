"""Tests for passive mDNS + DHCP discovery in actions/scanning.py.

Everything here is 100% offline: packets are crafted in-memory and fed to the
pure parsers, sockets/select/time are mocked. Nothing binds a port, sends a
probe or touches the network. Covers:

- mDNS DNS-name decoding (incl. compression pointers) and A/PTR parsing
- DHCP ACK / Request parsing and host->IP->MAC extraction
- pseudo-MAC derivation for MAC-less (mDNS) hosts
- accumulator merge semantics (real MAC wins, hostname enrichment)
- _passive_merge_to_db: real MAC, pseudo-MAC, existing-real-MAC reuse, blacklist
- passive_discover capture loop never transmits (asserts no sendto)
"""

import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from actions.scanning import NetworkScanner


# ---------------------------------------------------------------------------
# packet builders
# ---------------------------------------------------------------------------

def _encode_name(name):
    out = b''
    for label in name.split('.'):
        if not label:
            continue
        out += bytes([len(label)]) + label.encode()
    return out + b'\x00'


def _mdns_a_packet(name, ip, extra_answers=0):
    header = struct.pack('>HHHHHH', 0, 0x8400, 0, 1 + extra_answers, 0, 0)
    rdata = bytes(int(o) for o in ip.split('.'))
    answer = _encode_name(name) + struct.pack('>HHIH', 1, 0x8001, 120, 4) + rdata
    return header + answer


def _mdns_compressed_ptr_then_a(name, ip):
    """A packet where the A record's name is a compression pointer back to the
    PTR target, exercising _dns_read_name pointer handling."""
    header = struct.pack('>HHHHHH', 0, 0x8400, 0, 2, 0, 0)
    # answer 1: PTR  _svc.local -> name
    ptr_name = _encode_name('_svc._tcp.local')
    ptr_target_offset = 12 + len(ptr_name) + 10  # where target name begins
    ptr_rdata = _encode_name(name)
    a1 = ptr_name + struct.pack('>HHIH', 12, 0x8001, 120, len(ptr_rdata)) + ptr_rdata
    # answer 2: A record whose name points to the PTR target via compression
    pointer = 0xC000 | ptr_target_offset
    rdata = bytes(int(o) for o in ip.split('.'))
    a2 = struct.pack('>H', pointer) + struct.pack('>HHIH', 1, 0x8001, 120, 4) + rdata
    return header + a1 + a2


def _dhcp_packet(op, mac, msg_type, yiaddr='0.0.0.0', requested_ip=None,
                 ciaddr='0.0.0.0', hostname=None, good_cookie=True):
    mac_bytes = bytes(int(x, 16) for x in mac.split(':'))
    pkt = bytearray()
    pkt += bytes([op, 1, 6, 0])            # op, htype, hlen, hops
    pkt += struct.pack('>I', 0x12345678)   # xid
    pkt += struct.pack('>HH', 0, 0x8000)   # secs, flags (broadcast)
    pkt += bytes(int(o) for o in ciaddr.split('.'))   # ciaddr
    pkt += bytes(int(o) for o in yiaddr.split('.'))   # yiaddr
    pkt += b'\x00\x00\x00\x00'             # siaddr
    pkt += b'\x00\x00\x00\x00'             # giaddr
    pkt += mac_bytes + b'\x00' * (16 - len(mac_bytes))  # chaddr (16)
    pkt += b'\x00' * 64                     # sname
    pkt += b'\x00' * 128                    # file
    pkt += b'\x63\x82\x53\x63' if good_cookie else b'\x00\x00\x00\x00'
    # options
    pkt += bytes([53, 1, msg_type])
    if requested_ip is not None:
        pkt += bytes([50, 4]) + bytes(int(o) for o in requested_ip.split('.'))
    if hostname is not None:
        hb = hostname.encode()
        pkt += bytes([12, len(hb)]) + hb
    pkt += bytes([255])
    return bytes(pkt)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scanner():
    """A real NetworkScanner shell (no __init__) with just what the passive
    helpers touch."""
    s = NetworkScanner.__new__(NetworkScanner)
    s.logger = MagicMock()
    s.db = MagicMock()
    s.shared_data = SimpleNamespace(config={})
    s.blacklistcheck = False
    s.mac_scan_blacklist = set()
    s.ip_scan_blacklist = set()
    return s


class FakeSock:
    """A stand-in UDP socket that only yields queued packets. Any transmit
    attempt blows up the test, proving passive discovery never sends."""

    def __init__(self, port, packets):
        self._port = port
        self._packets = list(packets)
        self.closed = False

    def getsockname(self):
        return ('', self._port)

    def recvfrom(self, _bufsize):
        if self._packets:
            return self._packets.pop(0), ('0.0.0.0', self._port)
        raise BlockingIOError()

    def close(self):
        self.closed = True

    # Transmit methods must never be called.
    def sendto(self, *a, **k):   # pragma: no cover
        raise AssertionError("passive discovery must not transmit (sendto)")

    def send(self, *a, **k):     # pragma: no cover
        raise AssertionError("passive discovery must not transmit (send)")


# ---------------------------------------------------------------------------
# mDNS parsing
# ---------------------------------------------------------------------------

def test_dns_read_name_simple():
    buf = _encode_name('printer.local')
    name, nxt = NetworkScanner._dns_read_name(buf, 0)
    assert name == 'printer.local'
    assert nxt == len(buf)


def test_parse_mdns_a_record():
    pkt = _mdns_a_packet('nas.local', '192.168.1.50')
    parsed = NetworkScanner._parse_mdns_packet(pkt)
    assert ('nas.local', '192.168.1.50') in parsed['a_records']


def test_mdns_hosts_from_packet():
    pkt = _mdns_a_packet('tv.local', '192.168.1.77')
    hosts = NetworkScanner._mdns_hosts_from_packet(pkt)
    assert hosts == {'192.168.1.77': {'mac': '', 'vendor': '',
                                      'hostname': 'tv.local', 'source': 'mdns'}}


def test_mdns_compression_pointer():
    pkt = _mdns_compressed_ptr_then_a('cam.local', '10.0.0.9')
    parsed = NetworkScanner._parse_mdns_packet(pkt)
    assert ('cam.local', '10.0.0.9') in parsed['a_records']


def test_parse_mdns_garbage_is_safe():
    assert NetworkScanner._parse_mdns_packet(b'\x00\x01\x02') == {
        'a_records': [], 'ptr_records': []}
    assert NetworkScanner._parse_mdns_packet(b'') == {
        'a_records': [], 'ptr_records': []}


# ---------------------------------------------------------------------------
# DHCP parsing
# ---------------------------------------------------------------------------

def test_parse_dhcp_ack():
    pkt = _dhcp_packet(2, 'aa:bb:cc:dd:ee:ff', NetworkScanner.DHCP_ACK,
                       yiaddr='192.168.1.120', hostname='laptop')
    info = NetworkScanner._parse_dhcp_packet(pkt)
    assert info['msg_type'] == NetworkScanner.DHCP_ACK
    assert info['mac'] == 'aa:bb:cc:dd:ee:ff'
    assert info['yiaddr'] == '192.168.1.120'
    assert info['hostname'] == 'laptop'


def test_dhcp_host_from_ack():
    pkt = _dhcp_packet(2, 'aa:bb:cc:dd:ee:ff', NetworkScanner.DHCP_ACK,
                       yiaddr='192.168.1.120', hostname='laptop')
    ip, meta = NetworkScanner._dhcp_host_from_packet(pkt)
    assert ip == '192.168.1.120'
    assert meta == {'mac': 'aa:bb:cc:dd:ee:ff', 'vendor': '',
                    'hostname': 'laptop', 'source': 'dhcp'}


def test_dhcp_host_from_request_uses_requested_ip():
    pkt = _dhcp_packet(1, '11:22:33:44:55:66', NetworkScanner.DHCP_REQUEST,
                       requested_ip='192.168.1.200')
    ip, meta = NetworkScanner._dhcp_host_from_packet(pkt)
    assert ip == '192.168.1.200'
    assert meta['mac'] == '11:22:33:44:55:66'


def test_dhcp_bad_cookie_returns_none():
    pkt = _dhcp_packet(2, 'aa:bb:cc:dd:ee:ff', NetworkScanner.DHCP_ACK,
                       yiaddr='192.168.1.120', good_cookie=False)
    assert NetworkScanner._dhcp_host_from_packet(pkt) is None


def test_dhcp_zero_ip_returns_none():
    pkt = _dhcp_packet(1, 'aa:bb:cc:dd:ee:ff', NetworkScanner.DHCP_REQUEST)
    assert NetworkScanner._dhcp_host_from_packet(pkt) is None


# ---------------------------------------------------------------------------
# pseudo-MAC + accumulator merge
# ---------------------------------------------------------------------------

def test_pseudo_mac_for_ip():
    assert NetworkScanner._pseudo_mac_for_ip('192.168.1.50') == '00:00:c0:a8:01:32'
    assert NetworkScanner._pseudo_mac_for_ip('bad') == ''
    assert NetworkScanner._pseudo_mac_for_ip('1.2.3') == ''


def test_merge_passive_host_real_mac_wins_and_enriches():
    hosts = {}
    NetworkScanner._merge_passive_host(hosts, '10.0.0.1',
                                       {'mac': '', 'hostname': 'nas.local',
                                        'source': 'mdns'})
    NetworkScanner._merge_passive_host(hosts, '10.0.0.1',
                                       {'mac': 'aa:bb:cc:dd:ee:ff',
                                        'hostname': '', 'source': 'dhcp'})
    host = hosts['10.0.0.1']
    assert host['mac'] == 'aa:bb:cc:dd:ee:ff'
    assert host['hostname'] == 'nas.local'
    assert host['source'] == 'dhcp+mdns'


# ---------------------------------------------------------------------------
# _passive_merge_to_db
# ---------------------------------------------------------------------------

def test_merge_to_db_real_mac(scanner):
    hosts = {'10.0.0.5': {'mac': 'aa:bb:cc:dd:ee:ff', 'vendor': '',
                          'hostname': 'laptop', 'source': 'dhcp'}}
    written = scanner._passive_merge_to_db(hosts)
    assert written == 1
    scanner.db.upsert_host.assert_called_once()
    kwargs = scanner.db.upsert_host.call_args.kwargs
    assert kwargs['mac'] == 'aa:bb:cc:dd:ee:ff'
    assert kwargs['ip'] == '10.0.0.5'
    assert kwargs['hostname'] == 'laptop'
    scanner.db.add_scan_history.assert_called_once_with(
        'aa:bb:cc:dd:ee:ff', '10.0.0.5', 'passive_dhcp')


def test_merge_to_db_pseudo_mac_for_mdns(scanner):
    scanner.db.get_host_by_ip.return_value = None
    hosts = {'192.168.1.50': {'mac': '', 'vendor': '', 'hostname': 'nas.local',
                              'source': 'mdns'}}
    written = scanner._passive_merge_to_db(hosts)
    assert written == 1
    kwargs = scanner.db.upsert_host.call_args.kwargs
    assert kwargs['mac'] == '00:00:c0:a8:01:32'      # deterministic pseudo-MAC
    assert kwargs['hostname'] == 'nas.local'


def test_merge_to_db_reuses_existing_real_mac(scanner):
    scanner.db.get_host_by_ip.return_value = {'mac': 'de:ad:be:ef:00:01'}
    hosts = {'192.168.1.50': {'mac': '', 'vendor': '', 'hostname': 'nas.local',
                              'source': 'mdns'}}
    scanner._passive_merge_to_db(hosts)
    assert scanner.db.upsert_host.call_args.kwargs['mac'] == 'de:ad:be:ef:00:01'


def test_merge_to_db_blacklist_skips(scanner):
    scanner.blacklistcheck = True
    scanner.mac_scan_blacklist = {'aa:bb:cc:dd:ee:ff'}
    hosts = {'10.0.0.5': {'mac': 'aa:bb:cc:dd:ee:ff', 'source': 'dhcp'}}
    assert scanner._passive_merge_to_db(hosts) == 0
    scanner.db.upsert_host.assert_not_called()


# ---------------------------------------------------------------------------
# capture loop (mocked sockets/select/time) — proves it never transmits
# ---------------------------------------------------------------------------

def test_passive_discover_loop_is_passive(scanner, monkeypatch):
    import actions.scanning as scanning_mod

    mdns_pkt = _mdns_a_packet('tv.local', '192.168.1.77')
    dhcp_pkt = _dhcp_packet(2, 'aa:bb:cc:dd:ee:ff', NetworkScanner.DHCP_ACK,
                            yiaddr='192.168.1.120')
    mdns_sock = FakeSock(NetworkScanner.MDNS_PORT, [mdns_pkt])
    dhcp_sock = FakeSock(68, [dhcp_pkt])
    socks = [mdns_sock, dhcp_sock]

    # time: start, then one more read window, then past the deadline.
    times = iter([1000.0, 1000.1, 1000.2, 1000.3, 2000.0, 2000.0, 2000.0])
    monkeypatch.setattr(scanning_mod.time, 'time',
                        lambda: next(times, 3000.0))

    # select returns both sockets as readable on the first poll only.
    calls = {'n': 0}

    def fake_select(r, w, x, timeout):
        calls['n'] += 1
        return (list(r), [], []) if calls['n'] == 1 else ([], [], [])

    import select as select_mod
    monkeypatch.setattr(select_mod, 'select', fake_select)
    scanner.db.get_host_by_ip.return_value = None

    hosts = scanner.passive_discover(duration=1, _sockets=socks)

    assert '192.168.1.77' in hosts   # from mDNS
    assert '192.168.1.120' in hosts  # from DHCP
    # Injected sockets are not auto-closed (caller owns them); the point is the
    # loop read and merged without ever transmitting (FakeSock.sendto asserts).
    assert scanner.db.upsert_host.call_count == 2
