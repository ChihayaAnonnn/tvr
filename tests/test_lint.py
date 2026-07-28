"""Static checks that a three-hour training run should not have to discover.

hygiene_a0_seed0 trained all five epochs, wrote all five checkpoints, ran both
validation passes -- and then died on the final test with

    NameError: name 'selection_payload' is not defined

because a refactor moved the dict that built it into a helper and left the
reference behind. `ruff check --select F821` finds that in under a second. It
was already a project dependency; nothing ran it.

Only F821 is gated here. An undefined name is a guaranteed crash the moment
control reaches it, and in this codebase "the moment control reaches it" is
routinely hours of GPU time after launch. Unused imports and unused variables
are style, they do not end runs, and holding the suite hostage to them would
just get the whole check disabled the first time it was inconvenient.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_undefined_names():
    """No F821 anywhere ruff is configured to look.

    Failures print ruff's own concise output, which is already file:line:col
    plus the offending name -- enough to fix without rerunning anything.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "F821",
            "--output-format",
            "concise",
            ".",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "ruff found undefined names -- each one is a crash waiting for control "
        "flow to reach it:\n" + result.stdout + result.stderr
    )
