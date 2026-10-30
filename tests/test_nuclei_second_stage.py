"""Tests for the nuclei second stage in actions/nmap_vuln_scanner.py.

100% offline: nuclei and nmap are NEVER executed. subprocess is mocked and all
inputs are crafted strings. Covers:

- nmap -sV parsing into {port: service/version}
- nuclei -jsonl parsing (CVEs from tags + classification; malformed lines)
- port extraction from 'matched-at'
- CVE correlation by service/version
- target building (http/https/other) and command assembly
- binary resolution (config override + PATH)
- run_nuclei_stage: clear error when binary missing, success, timeout
- feeding findings to Network Intelligence + DB
- the guarded entry point (disabled / missing-binary paths never raise)
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import actions.nmap_vuln_scanner as mod
from actions.nmap_vuln_scanner import NmapVulnScanner, NucleiNotAvailableError


@pytest.fixture
def scanner():
    s = NmapVulnScanner.__new__(NmapVulnScanner)
    s.shared_data = SimpleNamespace(config={}, network_intelligence=None)
    s.db = MagicMock()
    return s


NMAP_SV = """
Starting Nmap 7.94
Nmap scan report for 10.0.0.5
Host is up (0.0010s latency).
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 9.2p1 Debian 2
80/tcp   open  http    Apache httpd 2.4.57
443/tcp  open  https   nginx 1.24.0
3306/tcp closed mysql
"""


def _nuclei_line(template_id, name, severity, host, port, cves=None,
                 cls_cves=None, matched_at=None):
    info = {"name": name, "severity": severity, "tags": list(cves or [])}
    if cls_cves is not None:
        info["classification"] = {"cve-id": cls_cves, "cvss-score": 7.5}
    obj = {"template-id": template_id, "info": info,
           "host": host, "matched-at": matched_at or f"{host}:{port}"}
    if port is not None:
        obj["port"] = port
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# parse_service_versions
# ---------------------------------------------------------------------------

def test_parse_service_versions(scanner):
    svc = scanner.parse_service_versions(NMAP_SV)
    assert set(svc) == {"22", "80", "443"}          # closed port dropped
    assert svc["22"]["service"] == "ssh"
    assert "OpenSSH 9.2p1" in svc["22"]["version"]
    assert svc["443"]["service"] == "https"
    assert svc["80"]["product"] == "Apache"


# ---------------------------------------------------------------------------
# parse_nuclei_jsonl
# ---------------------------------------------------------------------------

def test_parse_nuclei_jsonl_tags_and_classification(scanner):
    out = "\n".join([
        "[INF] some banner line, not JSON",
        _nuclei_line("apache-cve", "Apache RCE", "critical", "10.0.0.5", 80,
                     cves=["cve-2021-41773", "apache"]),
        "{ this is : not valid json",
        _nuclei_line("nginx-info", "nginx version", "info", "10.0.0.5", 443,
                     cls_cves=["CVE-2019-20372"]),
    ])
    findings = scanner.parse_nuclei_jsonl(out)
    assert len(findings) == 2
    f0 = findings[0]
    assert f0["cve_ids"] == ["CVE-2021-41773"]
    assert f0["severity"] == "critical"
    assert f0["port"] == "80"
    assert findings[1]["cve_ids"] == ["CVE-2019-20372"]


def test_parse_nuclei_jsonl_empty(scanner):
    assert scanner.parse_nuclei_jsonl("") == []
    assert scanner.parse_nuclei_jsonl(None) == []


# ---------------------------------------------------------------------------
# _port_from_matched_at
# ---------------------------------------------------------------------------

def test_port_from_matched_at(scanner):
    assert scanner._port_from_matched_at("https://10.0.0.5:8443/app") == "8443"
    assert scanner._port_from_matched_at("10.0.0.5:22") == "22"
    assert scanner._port_from_matched_at("http://10.0.0.5/") is None
    assert scanner._port_from_matched_at("[::1]:443") == "443"
    assert scanner._port_from_matched_at("") is None


# ---------------------------------------------------------------------------
# correlate_cves_by_service
# ---------------------------------------------------------------------------

def test_correlate_cves_by_service(scanner):
    port_services = scanner.parse_service_versions(NMAP_SV)
    findings = [
        {"template_id": "t1", "name": "n", "severity": "high",
         "cve_ids": ["CVE-2021-41773"], "port": "80", "matched_at": "",
         "cvss": 7.5},
        {"template_id": "t2", "name": "n2", "severity": "info",
         "cve_ids": [], "port": None, "matched_at": "https://10.0.0.5:443/",
         "cvss": None},
    ]
    corr = scanner.correlate_cves_by_service(findings, port_services)
    assert corr[0]["service"] == "http"
    assert "Apache httpd 2.4.57" in corr[0]["version"]
    assert corr[0]["cve_ids"] == ["CVE-2021-41773"]
    # second finding's port derived from matched-at -> 443 -> https
    assert corr[1]["port"] == "443"
    assert corr[1]["service"] == "https"


# ---------------------------------------------------------------------------
# targets + command
# ---------------------------------------------------------------------------

def test_nuclei_targets_from_services(scanner):
    svc = {"22": {"service": "ssh"}, "80": {"service": "http"},
           "443": {"service": "https"}}
    targets = scanner._nuclei_targets_from_services("10.0.0.5", svc)
    assert "https://10.0.0.5:443" in targets
    assert "http://10.0.0.5:80" in targets
    assert "10.0.0.5:22" in targets


def test_nuclei_targets_fallback_to_host(scanner):
    assert scanner._nuclei_targets_from_services("10.0.0.5", {}) == ["10.0.0.5"]


def test_build_nuclei_command(scanner):
    cmd = scanner.build_nuclei_command("/usr/bin/nuclei",
                                       ["http://10.0.0.5:80", "10.0.0.5:22"])
    assert cmd[0] == "/usr/bin/nuclei"
    assert "-jsonl" in cmd
    assert "-no-interactsh" in cmd
    assert cmd.count("-u") == 2
    assert "http://10.0.0.5:80" in cmd


# ---------------------------------------------------------------------------
# binary resolution
# ---------------------------------------------------------------------------

def test_find_nuclei_binary_config_override(scanner, monkeypatch):
    scanner.shared_data.config = {"nuclei_path": "/opt/nuclei"}
    monkeypatch.setattr(mod.os.path, "exists", lambda p: p == "/opt/nuclei")
    assert scanner._find_nuclei_binary() == "/opt/nuclei"


def test_find_nuclei_binary_from_path(scanner, monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda n: "/usr/bin/nuclei")
    assert scanner._find_nuclei_binary() == "/usr/bin/nuclei"


def test_find_nuclei_binary_absent(scanner, monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    assert scanner._find_nuclei_binary() is None


# ---------------------------------------------------------------------------
# run_nuclei_stage
# ---------------------------------------------------------------------------

def test_run_nuclei_stage_raises_when_missing(scanner, monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    with pytest.raises(NucleiNotAvailableError):
        scanner.run_nuclei_stage("10.0.0.5", {"80": {"service": "http"}})


def test_run_nuclei_stage_success(scanner, monkeypatch):
    svc = scanner.parse_service_versions(NMAP_SV)
    out = _nuclei_line("apache-cve", "Apache RCE", "critical", "10.0.0.5", 80,
                       cves=["cve-2021-41773"])
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout=out, stderr="", returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    corr = scanner.run_nuclei_stage("10.0.0.5", svc, nuclei_path="/usr/bin/nuclei")
    assert len(corr) == 1
    assert corr[0]["cve_ids"] == ["CVE-2021-41773"]
    assert corr[0]["service"] == "http"
    assert "/usr/bin/nuclei" in captured["cmd"]


def test_run_nuclei_stage_timeout_uses_partial(scanner, monkeypatch):
    out = _nuclei_line("t", "n", "high", "10.0.0.5", 80, cves=["cve-2020-1"])

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1),
                                        output=out.encode())

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    corr = scanner.run_nuclei_stage("10.0.0.5", {"80": {"service": "http"}},
                                    nuclei_path="/usr/bin/nuclei")
    assert len(corr) == 1
    assert corr[0]["cve_ids"] == ["CVE-2020-1"]


# ---------------------------------------------------------------------------
# feeding results
# ---------------------------------------------------------------------------

def test_feed_nuclei_findings(scanner):
    ni = MagicMock()
    scanner.shared_data.network_intelligence = ni
    correlated = [{
        "port": "80", "service": "http", "product": "Apache",
        "version": "Apache httpd 2.4.57", "cve_ids": ["CVE-2021-41773"],
        "severity": "critical", "template_id": "apache-cve", "name": "RCE",
        "cvss": 7.5, "source": "nuclei",
    }]
    scanner._feed_nuclei_findings("10.0.0.5", "web", "aa:bb:cc:dd:ee:ff",
                                  correlated)
    ni.add_vulnerability.assert_called_once()
    kw = ni.add_vulnerability.call_args.kwargs
    assert kw["vulnerability"] == "CVE-2021-41773"
    assert kw["port"] == 80
    scanner.db.upsert_host.assert_called_once()
    assert "CVE-2021-41773" in scanner.db.upsert_host.call_args.kwargs["vulnerabilities"]


# ---------------------------------------------------------------------------
# guarded entry point
# ---------------------------------------------------------------------------

def test_second_stage_disabled_returns_empty(scanner):
    scanner.shared_data.config = {}  # nuclei_stage_enabled absent -> off
    assert scanner._run_nuclei_second_stage("10.0.0.5", "h", "m", NMAP_SV) == []


def test_second_stage_missing_binary_is_swallowed(scanner, monkeypatch):
    scanner.shared_data.config = {"nuclei_stage_enabled": True}
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    # must not raise even though nuclei is absent
    assert scanner._run_nuclei_second_stage("10.0.0.5", "h", "m", NMAP_SV) == []
