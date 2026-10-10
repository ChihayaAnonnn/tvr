"""Strict, versioned evidence artifact schemas; legacy contracts stay untouched."""
from experiment_artifacts import artifact_schema_validator, validate_artifact_json_payload, artifact_json_bytes, strict_json_loads

DIGEST = {'type':'string','pattern':'^[0-9a-f]{64}$'}
STRING = {'type':'string','minLength':1}
NUMBER = {'type':'number'}
COUNT = {'type':'integer','minimum':0}


def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def array(items):
    return dict(type='array',items=items)


def schema_for(kind):
    ref = obj(dict(path=STRING,key=STRING,shape=array(COUNT),dtype=STRING,byte_size=COUNT,format={'const':'npy'},sha256=DIGEST))
    input_ref = {'anyOf':[{'type':'null'},obj(dict(path=STRING,sha256=DIGEST))]}
    parents = obj({k:input_ref for k in ('reference','matcher','value','calibration')})
    producer = obj({k:DIGEST for k in ('matcher_sha256','config_sha256','index_sha256','label_sha256','view_sha256','mc_sha256','state_sampling_sha256','feature_schema_sha256')})
    identity = obj(dict(producer_sha256=DIGEST,model_sha256=DIGEST,role={'const':'V_calibration'},relevance_space={'enum':['benchmark_annotation','reviewed_semantic']},feature_schema_sha256=DIGEST))
    resources = obj({k:COUNT for k in ('frame_requests','clip_frames','temporal_tokens','mc_elements')})
    schemas = {
        'input_failure': obj(dict(schema_version={'const':'evidence-input-failure-v1'},scope={'enum':['episode','index']},
            query_id=STRING,candidate_id=STRING,actual_counts=resources,
            budget={'anyOf':[{'type':'null'},obj(dict(limits=resources,spent=resources,reserved=resources))]},
            affected_query_count={'type':'integer','minimum':1},affected_query_ids_sha256=DIGEST,
            evaluation_complete={'const':False},overall_metrics_defined={'const':False})),
        'checkpoint_sidecar': obj(dict(schema_version={'const':'evidence-checkpoint-v1'},sha256=DIGEST,config_sha256=DIGEST,
            parents=parents,model_sha256={'type':'object','additionalProperties':DIGEST,'minProperties':1},stage=STRING,
            producer_sha256=DIGEST,world_size={'type':'integer','minimum':1},source_sha={'type':'string','pattern':'^[0-9a-f]{40}$'},manifest_sha256=DIGEST)),
        'transition_index': obj(dict(schema_version={'const':'evidence-transitions-v1'},producer=producer,arrays=array(ref),
            records=array(obj(dict(state_sha256=DIGEST,action_id=DIGEST,before_id=STRING,after_id=STRING,source_components=array(STRING)))),component_ids=array(STRING))),
        'transition_shard': obj(dict(schema_version={'const':'evidence-transition-shard-v1'},
            records=array(obj(dict(state_sha256=DIGEST,action_id=DIGEST,before_id=STRING,after_id=STRING,source_components=array(STRING)))),component_ids=array(DIGEST))),
        'transition_index_v2': obj(dict(schema_version={'const':'evidence-transitions-v2'},producer=producer,arrays=array(ref),
            record_count=COUNT,shards=array(obj(dict(path=STRING,sha256=DIGEST,count={'type':'integer','minimum':1}))))),
        'calibration': obj(dict(temperature={'type':'number','exclusiveMinimum':0},identity=identity,count={'type':'integer','minimum':1},schema_version={'const':'evidence-calibration-v1'})),
        'action_event': obj(dict(schema_version={'const':'evidence-action-v1'},action_id=DIGEST,event={'enum':['prepare','execute','commit','predict','predict_prepare']},
            payload={'type':'object'},payload_sha256=DIGEST)),
        'report': obj(dict(summary_metrics={'type':'object','additionalProperties':NUMBER},array_references=array(ref),
            undefined_reasons={'type':'object','additionalProperties':STRING},identity={'type':'object'})),
    }
    if kind not in schemas:
        raise ValueError('unknown evidence document kind')
    return schemas[kind]


def validate_evidence_document(kind,payload):
    validate_artifact_json_payload(payload)
    wire = strict_json_loads(artifact_json_bytes(payload).decode())
    errors = list(artifact_schema_validator(schema_for(kind)).iter_errors(wire))
    if errors:
        raise ValueError('invalid evidence document: '+errors[0].message)
