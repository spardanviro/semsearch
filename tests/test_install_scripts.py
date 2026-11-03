"""The one-line installers: what scripts/release-assets.sh relies on must stay in place."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_install_scripts_default_ref_line_is_present_once() -> None:
    # release-assets.sh rewrites exactly these lines; a renamed variable would ship release
    # installers that silently install `main`
    ps1 = (SCRIPTS / "install.ps1").read_text(encoding="utf-8")
    sh = (SCRIPTS / "install.sh").read_text(encoding="utf-8")

    assert len(re.findall(r"^    \$defaultRef = 'main'", ps1, re.M)) == 1
    assert len(re.findall(r"^DEFAULT_REF=main ", sh, re.M)) == 1


def test_install_scripts_never_exit_the_callers_powershell() -> None:
    # `irm | iex` runs the script in the user's own session: `exit` would close their window
    ps1 = (SCRIPTS / "install.ps1").read_text(encoding="utf-8")

    assert not re.search(r"^\s*exit\b", ps1, re.M)


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")
def test_release_assets_pin_the_tag(tmp_path: Path) -> None:
    out = tmp_path / "release"

    subprocess.run(["sh", str(SCRIPTS / "release-assets.sh"), "v9.8.7", str(out)], check=True,
                   capture_output=True)

    assert "$defaultRef = 'v9.8.7'" in (out / "install.ps1").read_text(encoding="utf-8")
    assert "DEFAULT_REF=v9.8.7 " in (out / "install.sh").read_text(encoding="utf-8")
    assert b"\r" not in (out / "install.sh").read_bytes()   # runs under Linux sh


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")
@pytest.mark.parametrize("tag", ["main", "v1.2", "v1.2.3;rm", "v1.2.3'x"])
def test_release_assets_reject_odd_tags(tmp_path: Path, tag: str) -> None:
    result = subprocess.run(["sh", str(SCRIPTS / "release-assets.sh"), tag, str(tmp_path)],
                            capture_output=True)

    assert result.returncode != 0
    assert not (tmp_path / "install.ps1").exists()
