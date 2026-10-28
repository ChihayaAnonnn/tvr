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


@pytest.mark.parametrize('scoped',[False,True])
def test_successor_import_restores_executed_action_without_repeating_io(journal_case,tmp_path,scoped):
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.recovery import export_recovery,import_recovery,save_initial_state,read_initial_state
    from semantic_retrieval.evidence.config import InputReference
    from semantic_retrieval.data.episodes import file_sha256
    c = journal_case
    if scoped:
        c.journal = ActionJournal(c.journal.root,c.journal.store,query_id=c.state.query_id)
    save_initial_state(c.journal.store,c.state)
    c.journal.prepare(c.state,c.decision)
    c.journal.record_execution(c.decision.action.action_id,c.outcome)
    path = export_recovery(c.journal.store,config_sha256='a'*64,source_sha='b'*40)
    successor = ArtifactArrayStore(tmp_path/'successor'/'arrays')
    ref = InputReference(path,file_sha256(path))
    import_recovery(ref,successor,config_sha256='a'*64,source_sha='b'*40)
    journal = ActionJournal(successor.run_root/'provenance'/'evidence-actions',successor,query_id=c.state.query_id if scoped else None)
    initial = read_initial_state(successor,c.state.query_id)
    assert initial == c.state
    assert journal.recover(initial) == c.next_state
    assert journal.recover(initial).budget.spent == c.next_state.budget.spent
    import_recovery(ref,successor,config_sha256='a'*64,source_sha='b'*40)
    with pytest.raises(ValueError,match='identity'):
        import_recovery(ref,successor,config_sha256='d'*64,source_sha='b'*40)


def test_recovery_export_rejects_nonprivate_hardlinked_artifact(tmp_path):
    import os
    import torch
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.recovery import export_recovery
    store=ArtifactArrayStore(tmp_path/'run'/'arrays')
    reference=store.write('private',torch.arange(4.))
    os.link(store.run_root/reference.path,tmp_path/'external-alias.npy')
    with pytest.raises(ValueError,match='private|unsafe'):
        export_recovery(store,config_sha256='a'*64,source_sha='b'*40)


@pytest.mark.parametrize('kind',['file-symlink','directory-symlink','fifo'])
def test_recovery_export_rejects_unsafe_tree_leaves(tmp_path,kind):
    import os
    import torch
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.recovery import export_recovery
    store = ArtifactArrayStore(tmp_path/'run'/'arrays')
    reference = store.write('private',torch.arange(4.))
    if kind == 'file-symlink':
        (store.root/'alias.npy').symlink_to(store.run_root/reference.path)
    elif kind == 'directory-symlink':
        (store.root/'alias').symlink_to(tmp_path,target_is_directory=True)
    else:
        os.mkfifo(store.root/'pipe.npy')
    with pytest.raises(ValueError,match='unsafe'):
        export_recovery(store,config_sha256='a'*64,source_sha='b'*40)


def test_recovery_export_rejects_directory_swap_during_inventory(tmp_path,monkeypatch):
    import os
    from semantic_retrieval.evidence.recovery import _directory_inventory
    root = tmp_path/'run'; directory = root/'arrays'; sub = directory/'sub'
    sub.mkdir(parents=True); (sub/'lost.npy').write_bytes(b'content')
    (directory/'keep.npy').write_bytes(b'other')
    original = os.stat; swapped = False
    def stat_then_swap(path,*args,**kwargs):
        nonlocal swapped
        value = original(path,*args,**kwargs)
        if path == 'sub' and kwargs.get('dir_fd') is not None and not swapped:
            swapped = True
            sub.rename(tmp_path/'moved')
            sub.symlink_to(tmp_path/'moved',target_is_directory=True)
        return value
    monkeypatch.setattr(os,'stat',stat_then_swap)
    with pytest.raises((ValueError,OSError)):
        _directory_inventory(directory,root)
    assert swapped


def test_scoped_journal_isolates_shared_video_predictions_and_recovers_legacy(state_factory,tmp_path):
    from dataclasses import replace
    import torch
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.journal import ActionJournal
    from semantic_retrieval.evidence.types import PairPredictiveScores,ResourceCounts,digest
    store=ArtifactArrayStore(tmp_path/'run'/'arrays');root=store.run_root/'provenance/evidence-actions'
    counts=ResourceCounts(0,0,4,8)
    def record(journal,query):
        before=state_factory(query_id=query)
        after=replace(before,budget=before.budget.reserve(counts).settle(counts,counts),version=1)
        scores=PairPredictiveScores(torch.full((2,),float(len(query))),torch.ones(2),torch.ones(2,4),torch.full((2,),-.6931472),tuple((query,c.canonical_id) for c in before.candidates),'a'*64)
        journal.prepare_prediction(before,counts);journal.record_prediction(before,after,scores,counts)
        return before,after
    legacy=ActionJournal(root,store);before,after=record(legacy,'legacy-query')
    scoped=ActionJournal(root,store,query_id='legacy-query')
    assert scoped.recover(before)==after
    assert scoped.prediction_for(after,expected_sampling='a'*64) is not None
    other=ActionJournal(root,store,query_id='other-query');other_before,other_after=record(other,'other-query')
    event=other._read(other_before.state_sha256,'predict')
    (store.run_root/event['arrays'][0]['path']).write_bytes(b'corrupt')
    assert scoped.recover(before)==after
    assert scoped.episode_history('legacy-query')[0]==after
    assert scoped.prediction_for(after,expected_sampling='a'*64) is not None
    with pytest.raises(ValueError):other.prediction_for(other_after,expected_sampling='a'*64)
    new=ActionJournal(root,store,query_id='new-query');new_before,new_after=record(new,'new-query')
    assert (root/digest('new-query')/(new_before.state_sha256+'.predict.json')).exists()
    with pytest.raises(ValueError):new.recover(before)
    reopened=ActionJournal(root,store,query_id='new-query')
    assert reopened.recover(new_before)==new_after
    with pytest.raises(ValueError,match='explicit query'):
        ActionJournal(root,store)
