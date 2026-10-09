"""Immutable prepare/execute/commit records; recovery never re-executes I/O."""
from dataclasses import asdict
import re

from experiment_artifacts import strict_json_load, verify_array_reference
from semantic_retrieval.evidence.frame_cache import publish_json, safe_path
from semantic_retrieval.evidence.policy import apply_outcome, reserve_action, validate_outcome
from semantic_retrieval.evidence.types import action_from_payload, digest, outcome_from_payload, state_from_payload


class ActionJournal:
    def __init__(self, root, store):
        if root != store.run_root/'provenance'/'evidence-actions':
            raise ValueError('journal must be run-local evidence-actions')
        safe_path(root)
        self.root = root
        self.store = store

    def _path(self, action_id, event):
        if not re.fullmatch('[0-9a-f]{64}', action_id) or event not in {'prepare', 'execute', 'commit', 'predict', 'predict_prepare'}:
            raise ValueError('invalid action event identity')
        p = self.root/(action_id+'.'+event+'.json')
        safe_path(p)
        return p

    def _write(self, action_id, event, payload):
        p = self._path(action_id,event)
        envelope = dict(schema_version='evidence-action-v1', action_id=action_id, event=event,
                        payload=payload, payload_sha256=digest(payload))
        from semantic_retrieval.evidence.schemas import validate_evidence_document
        validate_evidence_document('action_event',envelope)
        if p.exists():
            if digest(self._read(action_id,event)) != digest(payload):
                raise ValueError('immutable journal event conflict')
            return
        publish_json(p,envelope)

    def _read(self, action_id, event):
        p = self._path(action_id,event)
        if not p.exists():
            raise ValueError('missing action journal event; cannot establish execution')
        with p.open() as h:
            value = strict_json_load(h)
        from semantic_retrieval.evidence.schemas import validate_evidence_document
        validate_evidence_document('action_event',value)
        if set(value) != {'schema_version','action_id','event','payload','payload_sha256'} or value['schema_version'] != 'evidence-action-v1' or value['action_id'] != action_id or value['event'] != event or digest(value['payload']) != value['payload_sha256']:
            raise ValueError('corrupt journal event')
        return value['payload']

    def prepare(self, state, decision):
        expected = reserve_action(state, decision.action, predicted_counts=decision.action.reservation)
        if expected != decision:
            raise ValueError('journal reservation mismatch')
        self._write(decision.action.action_id,'prepare', dict(state=asdict(state), action=asdict(decision.action)))
        return decision.action.action_id

    def record_execution(self, action_id, outcome):
        prepared = self._read(action_id,'prepare')
        action = action_from_payload(prepared['action'])
        validate_outcome(action,outcome)
        for ref in outcome.feature_references:
            verify_array_reference(self.store.run_root,ref)
        self._write(action_id,'execute',asdict(outcome))

    def _next(self, action_id):
        prepared = self._read(action_id,'prepare')
        state = state_from_payload(prepared['state'])
        action = action_from_payload(prepared['action'])
        outcome = outcome_from_payload(self._read(action_id,'execute'))
        for ref in outcome.feature_references:
            verify_array_reference(self.store.run_root,ref)
        return state, apply_outcome(state,action,outcome)

    def commit(self, action_id, next_state):
        _, expected = self._next(action_id)
        if expected != next_state:
            raise ValueError('action commit state mismatch')
        self._write(action_id,'commit',dict(state_sha256=next_state.state_sha256))

    def prepare_prediction(self, before, upper):
        before.budget.reserve(upper)
        self._write(before.state_sha256,'predict_prepare',dict(before=asdict(before),upper=asdict(upper)))

    def record_prediction(self, before, after, scores, counts):
        from dataclasses import replace
        expected = replace(before,budget=before.budget.reserve(counts).settle(counts,counts),version=before.version+1)
        if expected != after:
            raise ValueError('prediction metering state mismatch')
        prepared = self._read(before.state_sha256,'predict_prepare')
        from semantic_retrieval.evidence.types import ResourceCounts
        if state_from_payload(prepared['before']) != before or not counts.fits(ResourceCounts(**prepared['upper'])):
            raise ValueError('prediction reservation mismatch')
        refs = [self.store.write('prediction_'+name,getattr(scores,name).detach().cpu()).as_dict()
                for name in ('loc','scale','features','ranking_log_probability')]
        self._write(before.state_sha256,'predict',dict(before=asdict(before),after=asdict(after),counts=asdict(counts),
                    arrays=refs,pair_ids=scores.pair_ids,sampling_sha256=scores.sampling_sha256))

    def _prediction(self, key):
        from dataclasses import replace
        from semantic_retrieval.evidence.types import ResourceCounts
        p = self._read(key,'predict')
        before,after = state_from_payload(p['before']),state_from_payload(p['after'])
        counts = ResourceCounts(**p['counts'])
        expected = replace(before,budget=before.budget.reserve(counts).settle(counts,counts),version=before.version+1)
        if expected != after or before.state_sha256 != key:
            raise ValueError('prediction journal state mismatch')
        for ref in p['arrays']:
            verify_array_reference(self.store.run_root,ref)
        return before,after,p

    def prediction_for(self,state,*,expected_sampling):
        import torch
        from semantic_retrieval.evidence.types import PairPredictiveScores
        if not self.root.exists():
            return None
        for path in self.root.glob('*.predict.json'):
            _,after,p = self._prediction(path.name.split('.')[0])
            if after == state:
                if p['sampling_sha256'] != expected_sampling:
                    raise ValueError('restored prediction model/integration mismatch')
                arrays = [torch.from_numpy(verify_array_reference(self.store.run_root,r).copy()) for r in p['arrays']]
                return PairPredictiveScores(*arrays,tuple(tuple(x) for x in p['pair_ids']),p['sampling_sha256'])
        return None

    def episode_history(self,query_id):
        predictions = [self._prediction(p.name.split('.')[0]) for p in self.root.glob('*.predict.json')]
        predictions = sorted((p for p in predictions if p[0].query_id == query_id),key=lambda p:p[0].version)
        actions = []
        for path in self.root.glob('*.prepare.json'):
            key = path.name.split('.')[0]
            state = state_from_payload(self._read(key,'prepare')['state'])
            if state.query_id == query_id:
                self._next(key)
                actions.append((state.version,key))
        return (predictions[0][1] if predictions else None),tuple(k for _,k in sorted(actions))

    def recover(self, state):
        if not self.root.exists():
            return state
        safe_path(self.root)
        history = []
        for p in self.root.glob('*.predict_prepare.json'):
            key = p.name.split('.')[0]
            before = state_from_payload(self._read(key,'predict_prepare')['before'])
            if before.query_id == state.query_id and not self._path(key,'predict').exists():
                raise ValueError('unfinished prediction; actual work cannot be established')
        for p in self.root.glob('*.prepare.json'):
            action_id = p.name.split('.')[0]
            prepared = self._read(action_id,'prepare')
            before = state_from_payload(prepared['state'])
            if before.query_id != state.query_id:
                continue
            before, after = self._next(action_id)
            if self._path(action_id,'commit').exists():
                if self._read(action_id,'commit') != {'state_sha256': after.state_sha256}:
                    raise ValueError('action commit digest mismatch')
            history.append((before,after,action_id))
        for p in self.root.glob('*.predict.json'):
            before,after,_ = self._prediction(p.name.split('.')[0])
            if before.query_id == state.query_id:
                history.append((before,after,None))
        history.sort(key=lambda item:item[0].version)
        for i, (before, after, action_id) in enumerate(history):
            if i and before != history[i-1][1]:
                raise ValueError('journal history is not a single verifiable state chain')
        valid_hashes = {s.state_sha256 for row in history for s in row[:2]}
        if history and state.state_sha256 not in valid_hashes:
            raise ValueError('checkpoint state is not in journal chain')
        for before, after, action_id in history:
            if state == before:
                if action_id is not None:
                    self.commit(action_id,after)
                state = after
        return state
