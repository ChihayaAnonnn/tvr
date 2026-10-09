import pytest


def test_recovery_rejects_corrupt_feature_reference(journal_case):
    c = journal_case
    c.journal.prepare(c.state, c.decision)
    c.journal.record_execution(c.decision.action.action_id, c.outcome)
    ref = c.outcome.feature_references[0]
    (c.journal.store.run_root/ref.path).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        c.journal.recover(c.state)


def test_prepare_without_execution_fails_closed(journal_case):
    c = journal_case; c.journal.prepare(c.state, c.decision)
    with pytest.raises(ValueError):
        c.journal.recover(c.state)
