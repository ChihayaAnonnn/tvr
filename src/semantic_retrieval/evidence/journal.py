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
        if not re.fullmatch('[0-9a-f]{64}', action_id) or event not in {'prepare', 'execute', 'commit'}:
            raise ValueError('invalid action event identity')
        p = self.root/(action_id+'.'+event+'.json')
        safe_path(p)
        return p

    def _write(self, action_id, event, payload):
        p = self._path(action_id,event)
        envelope = dict(schema_version='evidence-action-v1', action_id=action_id, event=event,
                        payload=payload, payload_sha256=digest(payload))
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

    def recover(self, state):
        if not self.root.exists():
            return state
        safe_path(self.root)
        history = []
        for p in self.root.glob('*.prepare.json'):
            action_id = p.name.split('.')[0]
            before, after = self._next(action_id)
            if self._path(action_id,'commit').exists():
                if self._read(action_id,'commit') != {'state_sha256': after.state_sha256}:
                    raise ValueError('action commit digest mismatch')
            history.append((before,after,action_id))
        history.sort(key=lambda item:item[0].version)
        for i, (before, after, action_id) in enumerate(history):
            if i and before != history[i-1][1]:
                raise ValueError('journal history is not a single verifiable state chain')
        valid_hashes = {s.state_sha256 for row in history for s in row[:2]}
        if history and state.state_sha256 not in valid_hashes:
            raise ValueError('checkpoint state is not in journal chain')
        for before, after, action_id in history:
            if state == before:
                self.commit(action_id,after)
                state = after
        return state
