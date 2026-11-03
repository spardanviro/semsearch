"""`semsearch doctor` and `semsearch download-model`: every check judged from plain facts, so
no test needs a GPU, the network or the real model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semsearch import cli, doctor
from semsearch.config import MODEL_ID, Settings
from semsearch.doctor import FAIL, OK, WARN, Check
from semsearch.install import Targets

GB = 1_000_000_000


# --- Python / uv ----------------------------------------------------------------------------
def test_doctor_python_too_old_fails_with_fix() -> None:
    check = doctor.judge_python((3, 11, 9))

    assert check.status == FAIL and "3.11.9" in check.detail and "3.12" in check.fix


def test_doctor_python_312_is_ok() -> None:
    assert doctor.judge_python((3, 12, 13)).status == OK


def test_doctor_missing_uv_is_only_a_warning() -> None:
    check = doctor.judge_uv(None)

    assert check.status == WARN and "astral.sh/uv" in check.fix
    assert doctor.judge_uv("/usr/bin/uv").status == OK


# --- GPU / PyTorch --------------------------------------------------------------------------
def test_doctor_gpu_in_use_is_ok() -> None:
    check = doctor.judge_gpu("13.0", True, "NVIDIA GeForce RTX 3060", "581.57")

    assert check.status == OK and "RTX 3060" in check.detail


def test_doctor_gpu_present_but_unused_because_driver_is_old_warns_about_the_driver() -> None:
    # the case README calls "silently falls back to the CPU"
    check = doctor.judge_gpu("13.0", False, None, "552.22")

    assert check.status == WARN
    assert "552.22" in check.detail and "580" in check.fix


def test_doctor_gpu_present_with_cpu_only_torch_warns_about_the_build() -> None:
    check = doctor.judge_gpu(None, False, None, "581.57")

    assert check.status == WARN and "CPU-only" in check.detail


def test_doctor_no_nvidia_gpu_runs_on_cpu_with_a_warning() -> None:
    check = doctor.judge_gpu("13.0", False, None, None)

    assert check.status == WARN and "CPU" in check.detail


def test_doctor_driver_version_parses_nvidia_smi_output() -> None:
    assert doctor.driver_major("581.57") == 581
    assert doctor.driver_major("garbage") is None


# --- model ----------------------------------------------------------------------------------
def test_doctor_cached_model_is_ok_without_network() -> None:
    check = doctor.judge_model(MODEL_ID, cached=True, remote="skipped", size=None)

    assert check.status == OK


@pytest.mark.parametrize(("remote", "status", "fix_has"), [
    ("ok", WARN, "semsearch download-model"),
    ("skipped", WARN, "semsearch download-model"),
    ("unreachable", FAIL, "HF_ENDPOINT"),
    ("gated", FAIL, "hf auth login"),
    ("missing", FAIL, "SEMSEARCH_MODEL"),
])
def test_doctor_model_not_cached_explains_how_to_get_it(remote: str, status: str,
                                                        fix_has: str) -> None:
    check = doctor.judge_model(MODEL_ID, cached=False, remote=remote, size=int(1.5 * GB))

    assert check.status == status and fix_has in check.fix


def test_doctor_model_not_cached_reports_its_download_size() -> None:
    check = doctor.judge_model(MODEL_ID, cached=False, remote="ok", size=int(1.5 * GB))

    assert "1.5 GB" in check.detail


class FlakyApi:
    """Fails the first `fails` calls like a dropped TLS connection, then answers."""

    def __init__(self, fails: int) -> None:
        self.fails = fails

    def model_info(self, *_a: Any, **_k: Any) -> Any:
        if self.fails:
            self.fails -= 1
            raise ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")
        return type("Info", (), {"siblings": [type("S", (), {"size": 1500})()]})()

    def auth_check(self, *_a: Any, **_k: Any) -> None:
        return None


def test_doctor_remote_model_retries_a_dropped_connection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FlakyApi(fails=1))

    assert doctor.remote_model(Settings()) == ("ok", 1500)


def test_doctor_remote_model_gives_up_after_repeated_failures(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FlakyApi(fails=99))

    assert doctor.remote_model(Settings()) == ("unreachable", None)


# --- disk / daemon --------------------------------------------------------------------------
def test_doctor_disk_too_small_for_the_model_fails() -> None:
    assert doctor.judge_disk(free=1 * GB, need=int(1.5 * GB)).status == FAIL
    assert doctor.judge_disk(free=50 * GB, need=int(1.5 * GB)).status == OK


def test_doctor_daemon_not_running_is_fine() -> None:
    assert doctor.judge_daemon(None).status == OK
    assert "pid=7" in doctor.judge_daemon({"pid": 7, "port": 9}).detail


# --- Claude Code registration ---------------------------------------------------------------
def _targets(tmp_path: Path) -> Targets:
    return Targets(tmp_path / ".claude.json", tmp_path / "settings.json", tmp_path / "agents",
                   tmp_path / "CLAUDE.md")


def _register(t: Targets, mcp: str) -> None:
    t.claude_json.write_text(json.dumps({"mcpServers": {"semsearch": {"command": mcp}}}),
                             encoding="utf-8")
    t.settings_json.write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Edit", "hooks": [{"type": "command", "command": '"x/semsearch-hook"'}]}]}}),
        encoding="utf-8")
    t.agents_dir.mkdir()
    (t.agents_dir / "semantic-search.md").write_text("agent", encoding="utf-8")
    t.claude_md.write_text("<!-- semsearch:start -->\nrules\n<!-- semsearch:end -->\n",
                           encoding="utf-8")


def test_doctor_claude_not_registered_suggests_install_claude(tmp_path: Path) -> None:
    check = doctor.judge_claude(_targets(tmp_path), expected_mcp=None)

    assert check.status == WARN and "semsearch install-claude" in check.fix


def test_doctor_claude_fully_registered_is_ok(tmp_path: Path) -> None:
    mcp = tmp_path / "semsearch-mcp.exe"
    mcp.write_text("", encoding="utf-8")
    t = _targets(tmp_path)
    _register(t, str(mcp))

    assert doctor.judge_claude(t, expected_mcp=str(mcp)).status == OK


def test_doctor_claude_registration_pointing_at_a_deleted_program_fails(tmp_path: Path) -> None:
    t = _targets(tmp_path)
    _register(t, str(tmp_path / "gone" / "semsearch-mcp.exe"))

    check = doctor.judge_claude(t, expected_mcp=None)

    assert check.status == FAIL and "semsearch install-claude" in check.fix


def test_doctor_claude_partial_registration_lists_missing_pieces(tmp_path: Path) -> None:
    mcp = tmp_path / "semsearch-mcp.exe"
    mcp.write_text("", encoding="utf-8")
    t = _targets(tmp_path)
    _register(t, str(mcp))
    t.settings_json.write_text("{}", encoding="utf-8")

    check = doctor.judge_claude(t, expected_mcp=str(mcp))

    assert check.status == WARN and "hook" in check.detail


def test_doctor_claude_unreadable_config_is_reported_not_raised(tmp_path: Path) -> None:
    t = _targets(tmp_path)
    t.claude_json.write_text("{not json", encoding="utf-8")

    check = doctor.judge_claude(t, expected_mcp=None)

    assert check.status == WARN


# --- report / runner ------------------------------------------------------------------------
def test_doctor_report_shows_fixes_and_exit_code_follows_failures() -> None:
    checks = [Check("Python", OK, "3.12"), Check("Model", FAIL, "not here", "do this")]

    text = doctor.format_report(checks)

    assert "do this" in text and "Python" in text
    assert doctor.exit_code(checks) == 1
    assert doctor.exit_code(checks[:1]) == 0


def test_doctor_a_crashing_check_becomes_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_ctx: doctor.Context) -> Check:
        raise RuntimeError("kaput")

    monkeypatch.setattr(doctor, "CHECKS", (("Broken", boom),))

    checks = doctor.run_checks(online=False)

    assert checks == [Check("Broken", FAIL, "check failed: RuntimeError: kaput",
                            "please report this at the project's issue tracker")]


def test_doctor_bad_settings_are_reported_instead_of_crashing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMSEARCH_EMBED_DIM", "not-a-number")
    monkeypatch.setattr(doctor, "CHECKS", ())

    checks = doctor.run_checks(online=False)

    assert [c.status for c in checks] == [FAIL] and "SEMSEARCH_" in checks[0].fix


def test_cli_doctor_offline_json(monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(doctor, "run_checks",
                        lambda online: seen.append(online) or [Check("Python", OK, "3.12")])

    code = cli.main(["doctor", "--offline", "--json"])

    assert code == 0 and seen == [False]
    assert json.loads(capsys.readouterr().out)[0]["status"] == OK


# --- download-model -------------------------------------------------------------------------
class FakeHub:
    def __init__(self, cached: bool = False, size: int | None = int(1.5 * GB)) -> None:
        self.cached, self.size, self.downloads = cached, size, 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "model_cached", lambda settings: self.cached)
        monkeypatch.setattr(doctor, "remote_model", lambda settings: ("ok", self.size))
        monkeypatch.setattr(doctor, "fetch_model", self.fetch)
        monkeypatch.setattr(doctor, "free_bytes", lambda path: 100 * GB)

    def fetch(self, settings: Settings) -> str:
        self.downloads += 1
        return "/cache/model"


def test_download_model_asks_first_and_does_nothing_on_no(
        monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)
    asked: list[str] = []

    code = doctor.download_model(Settings(), yes=False, ask=lambda q: asked.append(q) or "n")

    assert code == 1 and hub.downloads == 0
    assert "1.5 GB" in asked[0]


def test_download_model_downloads_after_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)

    assert doctor.download_model(Settings(), yes=False, ask=lambda q: "y") == 0
    assert hub.downloads == 1


def test_download_model_skips_when_already_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub(cached=True)
    hub.install(monkeypatch)

    assert doctor.download_model(Settings(), yes=True, ask=lambda q: "y") == 0
    assert hub.downloads == 0


def test_download_model_without_a_terminal_counts_as_no(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)

    def no_tty(_q: str) -> str:
        raise EOFError

    assert doctor.download_model(Settings(), yes=False, ask=no_tty) == 1
    assert hub.downloads == 0


def test_download_model_refuses_when_the_disk_is_too_small(
        monkeypatch: pytest.MonkeyPatch) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)
    monkeypatch.setattr(doctor, "free_bytes", lambda path: 1 * GB)

    assert doctor.download_model(Settings(), yes=True, ask=lambda q: "y") == 1
    assert hub.downloads == 0


def test_cli_download_model_passes_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(doctor, "download_model",
                        lambda settings, yes, ask: calls.append(yes) or 0)

    assert cli.main(["download-model", "--yes"]) == 0
    assert calls == [True]


# --- fact gathering (no GPU, network or model: each source is replaced) ---------------------
def _ctx(online: bool = False) -> doctor.Context:
    return doctor.Context(Settings(), online)


def test_doctor_check_model_cached_skips_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "model_cached", lambda s: True)
    monkeypatch.setattr(doctor, "remote_model", lambda s: pytest.fail("network used"))

    assert doctor.check_model(_ctx(online=True)).status == OK


def test_doctor_check_model_offline_does_not_ask_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "model_cached", lambda s: False)
    monkeypatch.setattr(doctor, "remote_model", lambda s: pytest.fail("network used"))

    assert doctor.check_model(_ctx(online=False)).status == WARN


def test_doctor_check_model_online_reports_the_hub_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "model_cached", lambda s: False)
    monkeypatch.setattr(doctor, "remote_model", lambda s: ("unreachable", None))

    assert doctor.check_model(_ctx(online=True)).status == FAIL


def test_doctor_checks_needing_settings_are_skipped_when_settings_are_invalid() -> None:
    ctx = doctor.Context(None, online=True)

    assert doctor.check_model(ctx).status == WARN
    assert doctor.check_daemon(ctx).status == WARN


def test_doctor_check_disk_needs_room_only_for_a_missing_model(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "free_bytes", lambda p: 1 * GB)
    monkeypatch.setattr(doctor, "model_cached", lambda s: True)
    assert doctor.check_disk(_ctx()).status == OK

    monkeypatch.setattr(doctor, "model_cached", lambda s: False)
    assert doctor.check_disk(_ctx()).status == FAIL


def test_doctor_free_bytes_uses_the_nearest_existing_folder(tmp_path: Path) -> None:
    assert doctor.free_bytes(tmp_path / "not" / "yet" / "created") > 0


def test_doctor_check_daemon_never_starts_one(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[bool] = []

    class Client:
        def __init__(self, _settings: Settings, autostart: bool) -> None:
            made.append(autostart)

        def running(self) -> None:
            return None

    monkeypatch.setattr(doctor, "DaemonClient", Client)

    assert doctor.check_daemon(_ctx()).status == OK and made == [False]


def test_doctor_check_claude_reads_the_user_files(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    monkeypatch.setattr(Targets, "default", classmethod(lambda cls: _targets(tmp_path)))

    def no_script(_name: str) -> str:
        raise FileNotFoundError

    monkeypatch.setattr(doctor, "script_path", no_script)

    assert doctor.check_claude(_ctx()).detail == "semsearch is not registered"


def test_doctor_check_gpu_combines_torch_and_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "nvidia_driver", lambda: "552.22")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    assert "too old" in doctor.check_gpu(_ctx()).detail


def test_doctor_nvidia_driver_reads_the_first_gpu(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    import subprocess
    smi = tmp_path / "System32" / "nvidia-smi.exe"
    smi.parent.mkdir()
    smi.write_text("", encoding="utf-8")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: str(smi))
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="581.57\n581.57\n"))

    assert doctor.nvidia_driver() == "581.57"


def test_doctor_nvidia_driver_failure_means_no_gpu(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    import subprocess
    smi = tmp_path / "System32" / "nvidia-smi.exe"
    smi.parent.mkdir()
    smi.write_text("", encoding="utf-8")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: str(smi))

    def broken(*_a: Any, **_k: Any) -> Any:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(doctor.subprocess, "run", broken)

    assert doctor.nvidia_driver() is None


def test_doctor_nvidia_driver_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)

    assert doctor.nvidia_driver() is None


def test_download_model_stops_when_the_hub_refuses(monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)
    monkeypatch.setattr(doctor, "remote_model", lambda s: ("gated", None))

    assert doctor.download_model(Settings(), yes=True, ask=lambda q: "y") == 1
    assert "hf auth login" in capsys.readouterr().out and hub.downloads == 0


def test_download_model_failure_suggests_a_mirror(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    hub = FakeHub()
    hub.install(monkeypatch)

    def broken(_s: Settings) -> str:
        raise OSError("connection reset")

    monkeypatch.setattr(doctor, "fetch_model", broken)

    assert doctor.download_model(Settings(), yes=True, ask=lambda q: "y") == 1
    assert "HF_ENDPOINT" in capsys.readouterr().out


def _fake_snapshot(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(folder))


def test_doctor_interrupted_download_is_not_counted_as_cached(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # the snapshot folder exists with the small files, but the weights never arrived
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _fake_snapshot(monkeypatch, tmp_path)

    assert doctor.model_cached(Settings()) is False


def test_doctor_complete_snapshot_is_cached(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"w")
    _fake_snapshot(monkeypatch, tmp_path)

    assert doctor.model_cached(Settings()) is True


def test_doctor_claude_bare_command_name_is_not_called_missing(tmp_path: Path) -> None:
    t = _targets(tmp_path)
    _register(t, "semsearch-mcp")

    assert doctor.judge_claude(t, expected_mcp=None).status == OK


def test_cli_download_model_reports_bad_settings(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SEMSEARCH_EMBED_DIM", "not-a-number")

    assert cli.main(["download-model"]) == 2
    assert "SEMSEARCH_EMBED_DIM" in capsys.readouterr().err
