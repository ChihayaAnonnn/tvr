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

def test_journal_keeps_query_histories_separate(journal_case):
    from dataclasses import replace
    from semantic_retrieval.evidence.views import propose_actions
    from semantic_retrieval.evidence.policy import reserve_action,apply_outcome
    c = journal_case
    c.journal.prepare(c.state,c.decision)
    c.journal.record_execution(c.decision.action.action_id,c.outcome)
    c.journal.commit(c.decision.action.action_id,c.next_state)
    other = replace(c.state,query_id='other-query')
    action = propose_actions(other)[0]
    decision = reserve_action(other,action,predicted_counts=action.reservation)
    outcome = replace(c.outcome,action_id=action.action_id)
    c.journal.prepare(other,decision);c.journal.record_execution(action.action_id,outcome)
    assert c.journal.recover(other) == apply_outcome(other,action,outcome)

def test_prediction_metering_restores_scores_without_new_cost(journal_case):
    from dataclasses import replace
    import torch
    from semantic_retrieval.evidence.types import PairPredictiveScores,ResourceCounts
    c = journal_case
    scores = PairPredictiveScores(torch.zeros(2),torch.ones(2),torch.ones(2,4),torch.full((2,),-.6931472),
                                  (('q0','a'),('q0','b')),'a'*64)
    counts = ResourceCounts(0,0,4,8)
    after = replace(c.state,budget=c.state.budget.reserve(counts).settle(counts,counts),version=1)
    c.journal.prepare_prediction(c.state,counts)
    c.journal.record_prediction(c.state,after,scores,counts)
    assert c.journal.recover(c.state) == after
    restored = c.journal.prediction_for(after,expected_sampling='a'*64)
    assert torch.equal(restored.loc,scores.loc)
    assert c.journal.recover(after) == after


def test_unfinished_prediction_fails_closed_for_its_query_only(journal_case):
    from dataclasses import replace
    from semantic_retrieval.evidence.types import ResourceCounts
    c = journal_case
    c.journal.prepare_prediction(c.state,ResourceCounts(0,0,4,8))
    with pytest.raises(ValueError,match='unfinished prediction'):
        c.journal.recover(c.state)
    other = replace(c.state,query_id='other-query')
    assert c.journal.recover(other) == other


def test_successor_import_restores_executed_action_without_repeating_io(journal_case,tmp_path):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.recovery import export_recovery,import_recovery,save_initial_state,read_initial_state
    from semantic_retrieval.evidence.config import InputReference
    from semantic_retrieval.data.episodes import file_sha256
    c = journal_case
    save_initial_state(c.journal.store,c.state)
    c.journal.prepare(c.state,c.decision)
    c.journal.record_execution(c.decision.action.action_id,c.outcome)
    path = export_recovery(c.journal.store,config_sha256='a'*64,source_sha='b'*40)
    successor = ArtifactArrayStore(tmp_path/'successor'/'arrays')
    ref = InputReference(path,file_sha256(path))
    import_recovery(ref,successor,config_sha256='a'*64,source_sha='b'*40)
    journal = ActionJournal(successor.run_root/'provenance'/'evidence-actions',successor)
    initial = read_initial_state(successor,c.state.query_id)
    assert initial == c.state
    assert journal.recover(initial) == c.next_state
    assert journal.recover(initial).budget.spent == c.next_state.budget.spent
    import_recovery(ref,successor,config_sha256='a'*64,source_sha='b'*40)
    with pytest.raises(ValueError,match='identity'):
        import_recovery(ref,successor,config_sha256='d'*64,source_sha='b'*40)
