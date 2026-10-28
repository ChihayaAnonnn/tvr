import pytest
from scripts.train_evidence_retrieval import main


def test_cli_has_no_output_or_gate_bypass_switches():
    with pytest.raises(SystemExit):
        main(['--output-dir','/tmp/unsafe'])
