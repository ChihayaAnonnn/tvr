"""Approved source components and deterministic, outcome-independent pilot roles."""
import hashlib
from collections import defaultdict
from urllib.parse import parse_qs, urlparse

from semantic_retrieval.evidence.source_audit import summarize_sources


def _order(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest(), value


def partition_sources(rows, train, validation, content, *, seed):
    """Union source/content edges; quarantine train components touching validation.

    Return video->role, video->component, excluded training IDs. Component IDs
    include quarantined members. Role boundaries never depend on labels/outcomes.
    """
    summarize_sources(rows, train, validation, content)  # Validate the complete inventory.
    parent = {i:i for i in train | validation}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    sources, hashes = {}, {}
    for row in sorted(rows, key=lambda r:r['video_id']):
        i = row['video_id']
        source = parse_qs(urlparse(row['url']).query)['v'][0]
        for table, key in ((sources, source), (hashes, content[i])):
            if key in table:
                a,b = sorted((root(i),root(table[key])))
                parent[b] = a
            else:
                table[key] = i
    members = defaultdict(list)
    for i in sorted(parent):
        members[root(i)].append(i)
    components, groups, excluded = {}, {}, set()
    import json
    for ids in members.values():
        component = hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        components.update({i:component for i in ids})
        groups[component] = ids
        if set(ids) & validation:
            excluded.update(set(ids) & train)
    training = sorted((c for c,ids in groups.items() if not set(ids)&validation),key=lambda c:_order(seed,c))
    valid = sorted((c for c,ids in groups.items() if set(ids)&validation),key=lambda c:_order(seed,c))
    roles = {}
    for index,c in enumerate(training):
        role = 'T_match' if index < len(training)*8//10 else 'T_value'
        roles.update({i:role for i in groups[c]})
    for index,c in enumerate(valid):
        role = ('V_development' if index < len(valid)*6//10 else
                'V_calibration' if index < len(valid)*8//10 else 'V_locked')
        roles.update({i:role for i in groups[c] if i in validation})
    return roles,components,excluded


def select_pilot(roles, components, *, per_role, seed):
    """Choose distinct components, then one video per component; no label inputs."""
    if type(per_role) is not int or per_role < 1 or not roles or not set(roles)<=components.keys():
        raise ValueError('invalid pilot membership/count')
    grouped, ownership = defaultdict(lambda:defaultdict(list)), {}
    for i,role in roles.items():
        c = components[i]
        if c in ownership and ownership[c] != role:
            raise ValueError('component crosses roles')
        ownership[c] = role
        grouped[role][c].append(i)
    result = {}
    for role,groups in sorted(grouped.items()):
        if len(groups) < per_role:
            raise ValueError('insufficient independent source components')
        chosen = sorted(groups,key=lambda c:_order(seed,c))[:per_role]
        result[role] = [min(groups[c],key=lambda i:_order(seed,i)) for c in chosen]
    return result


def preparation_spec():
    return dict(schema_version='evidence-source-input-v1',
                source_policy='youtube-id-content-sha256-v1', role_seed=0, pilot_seed=713,
                pilot_components_per_role=64, expected_excluded_train=254, expected_eligible_train=8246,
                annotation='/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json',
                annotation_sha256='8342df0ed6f8212ec8f4363e7415ce9a98d6ddf89e25ed2fb72b364f97c65d33',
                allowlist='/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv',
                allowlist_sha256='0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3',
                trusted_split='dataloaders/splits/msrvtt_trusted_v1_seed0.json',
                trusted_split_sha256='a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882',
                video_directory='/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps',
                caption_relation_sha256='bc2c06cbbe833d4e504700430a8a96c23bb7a79f2a5418e1a137ab0fe32b123d',
                gallery_version='evidence-msrvtt-source-pilot-v1',
                clip_sha256='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
                views=dict(pool_size=16,levels=[4,8,16],
                           preprocessing_sha256='8fddcf29db23466cb3d33d41a83510b8b6d9ad29bc56f4da634d02d6993a7972',
                           decode_protocol='opencv-rgb-v1'))


def preparation_scenario():
    from semantic_retrieval.evidence.types import digest
    return 'evidence_sources:'+digest(preparation_spec())


def export_pilot(root, spec, captions, selected, components, content, frame_counts, source_report):
    """Write bounded benchmark manifests, then validate the real production reader.

    Episode candidate columns are label-independent metadata shards. Their union
    defines the full gallery; the worker builds its sparse CLIP index before
    binding query labels. No positive is inserted into the retrieved top K.
    """
    from pathlib import Path
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.evidence.config import InputReference, SplitBinding
    from semantic_retrieval.evidence.data import read_split_binding, audit_component_roles
    from semantic_retrieval.evidence.frame_cache import publish_json
    from semantic_retrieval.evidence.types import digest
    from semantic_retrieval.runtime.input_bindings import write_input_indices
    root = Path(root)
    def put(path, value):
        path = root/path
        if path.exists():
            raise FileExistsError(path)
        publish_json(path,value)
        return dict(path=str(path),sha256=file_sha256(path))
    all_ids = {i for ids in selected.values() for i in ids}
    chosen = {}
    for row in captions:
        i = row['official_video_id']
        if i in all_ids and (i not in chosen or _order(spec['pilot_seed'],row['query_id']) < _order(spec['pilot_seed'],chosen[i]['query_id'])):
            chosen[i] = row
    if set(chosen) != all_ids or not all_ids<=frame_counts.keys():
        raise ValueError('missing selected captions or decoded frame counts')
    annotations = put('pilot-annotations.json',dict(schema_version=1,rows=[chosen[i] for i in sorted(chosen)]))
    report = put('source-report.json',source_report)
    bindings, references = [], [annotations,report]
    split_names = dict(T_match='train',T_value='train',V_development='val_development',V_calibration='val_calibration',V_locked='val_locked')
    for role,ids in sorted(selected.items()):
        ids = sorted(ids)
        relation = put('labels/'+role+'-relations.json',{chosen[i]['query_id']:[i] for i in ids})
        id_map = put('labels/'+role+'-id-map.json',{i:'msrvtt:'+i for i in ids})
        protocol = dict(protocol_id='msrvtt-source-pilot-benchmark-v1',annotation_sha256=annotations['sha256'],
                        relation_sha256=relation['sha256'],id_map_sha256=id_map['sha256'],source_report_sha256=report['sha256'],
                        allowlist_sha256=spec['allowlist_sha256'],source_annotation_sha256=spec['annotation_sha256'])
        episodes = []
        for n,i in enumerate(ids):
            # Every inventory shard is repeated deterministically, independent of the query's positive.
            shard = ids[(n%((len(ids)+7)//8))*8:((n%((len(ids)+7)//8))+1)*8]
            canonical = ['msrvtt:'+j for j in shard]
            episodes.append(dict(query_id=chosen[i]['query_id'],query_text=chosen[i]['query_text'],gallery_version=spec['gallery_version'],
                candidate_ids=canonical,canonical_ids=canonical,relevance=['positive' if j==i else 'negative' for j in shard],
                answer_in_candidates='yes' if i in shard else 'no',answer_in_gallery='yes',
                annotation_provenance=dict(source='benchmark_annotation',sha256=annotations['sha256'],protocol_id=protocol['protocol_id']),
                split=split_names[role],video_paths=[str(Path(spec['video_directory'])/(j+'.mp4')) for j in shard],
                content_sha256=[content[j] for j in shard],frame_counts=[frame_counts[j] for j in shard],
                source_ids=[components[j] for j in shard],caption_group_id=components[i],
                query_source_ids=[components[i]],query_component_id=components[i]))
        payload = dict(schema_version=2,gallery_version=spec['gallery_version'],
                       preprocessing=dict(max_text_length=77,max_frames=16,image_size=224),episodes=episodes,
                       relevance_space='benchmark_annotation',label_protocol=protocol,
                       label_references=dict(relations=dict(path='../labels/'+role+'-relations.json',sha256=relation['sha256']),
                                             id_map=dict(path='../labels/'+role+'-id-map.json',sha256=id_map['sha256'])))
        manifest = put('roles/'+role+'.json',payload)
        binding = SplitBinding(role,InputReference(Path(manifest['path']),manifest['sha256']),'benchmark_annotation')
        loaded = read_split_binding(binding)
        if {c for e in loaded for c in e.episode.canonical_ids} != set(canonical_id for canonical_id in ('msrvtt:'+j for j in ids)):
            raise ValueError('pilot gallery coverage incomplete')
        bindings.append(binding);references.extend((relation,id_map,manifest))
    audit_component_roles(bindings)
    index = put('initial-index.json',dict(schema_version='evidence-initial-index-v1',clip_sha256=spec['clip_sha256'],
                preprocessing_sha256=spec['views']['preprocessing_sha256'],view_sha256=digest(spec['views']),
                gallery_version=spec['gallery_version'],roles={role:sorted('msrvtt:'+i for i in ids) for role,ids in selected.items()}))
    references.append(index)
    video_refs = [dict(path=str(Path(spec['video_directory'])/(i+'.mp4')),sha256=content[i]) for i in sorted(all_ids)]
    references.extend(write_input_indices(root,'pilot-videos',video_refs))
    return dict(splits=[dict(role=b.role,manifest=dict(path=str(b.manifest.path),sha256=b.manifest.sha256),
                             relevance_space=b.relevance_space) for b in bindings],index=index,input_references=references,
                roles_disjoint=True,gallery_coverage_verified=True,selected_video_count=len(all_ids))


def prepare_sources(context):
    import csv
    import json
    import time
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.data.msrvtt_snapshot import training_annotations, caption_relation_digest, inspect_training_video
    from semantic_retrieval.evidence.source_audit import training_source_metadata
    from semantic_retrieval.evidence.frame_cache import publish_json
    from semantic_retrieval.runtime.input_bindings import context_input_references
    spec = preparation_spec(); started = time.monotonic()
    references = context_input_references(context)
    paths = {}
    for name in ('annotation','allowlist','trusted_split'):
        path = Path(spec[name]); path = path if path.is_absolute() else context.repo_root/path
        if references.get(str(path),references.get(spec[name])) != spec[name+'_sha256'] or file_sha256(path) != spec[name+'_sha256']:
            raise ValueError('source preparation input not bound to its approved bytes')
        paths[name] = path
    with paths['allowlist'].open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['video_id']:
            raise ValueError('invalid training allowlist schema')
        ids = [r['video_id'] for r in reader]
    trusted = json.loads(paths['trusted_split'].read_bytes())
    train,val = set(trusted['train_video_ids']),set(trusted['val_video_ids'])
    if len(ids)!=9000 or len(set(ids))!=9000 or len(train)!=8500 or len(val)!=500 or train&val or train|val!=set(ids):
        raise ValueError('trusted membership changed')
    raw = paths['annotation'].read_bytes()
    rows,metadata_privacy = training_source_metadata(raw,set(ids))
    captions,caption_privacy = training_annotations(raw,set(ids))
    if caption_relation_digest(captions) != spec['caption_relation_sha256']:
        raise ValueError('published training caption relation changed')
    content = {}
    for i in sorted(ids):
        path = Path(spec['video_directory'])/(i+'.mp4')
        sha = references.get(str(path))
        if sha is None or file_sha256(path) != sha:
            raise ValueError('source video missing from reserved inventory or changed')
        content[i] = sha
    roles,components,excluded = partition_sources(rows,train,val,content,seed=spec['role_seed'])
    if len(excluded)!=spec['expected_excluded_train'] or len(train-excluded)!=spec['expected_eligible_train']:
        raise ValueError('approved quarantine cardinality changed')
    selected = select_pilot(roles,components,per_role=spec['pilot_components_per_role'],seed=spec['pilot_seed'])
    out = context.paths.provenance/'inputs';out.mkdir()
    assignment = [dict(video_id=i,component=components[i],role=roles.get(i,'quarantined_train')) for i in sorted(components)]
    for n in range(0,len(assignment),1000):
        publish_json(out/f'role-assignment-{n//1000:02d}.json',dict(schema_version=1,rows=assignment[n:n+1000]))
    cv2.setNumThreads(1)
    def inspect(i):
        return i,inspect_training_video(Path(spec['video_directory'])/(i+'.mp4'),expected_sha256=content[i],image_size=224,max_frames=16)
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = dict(pool.map(inspect,sorted(i for values in selected.values() for i in values)))
    report = dict(schema_version=spec['schema_version'],source_sha=context.config.source_sha,
                  manifest_sha256=context.config.manifest_sha256,source_definition=spec['source_policy'],
                  original_train_count=len(train),validation_pool_count=len(val),excluded_train_count=len(excluded),
                  eligible_train_count=len(train-excluded),role_video_counts={r:sum(v==r for v in roles.values()) for r in selected},
                  role_component_counts={r:len({components[i] for i in roles if roles[i]==r}) for r in selected},
                  pilot_components_per_role=spec['pilot_components_per_role'],pilot_video_count=len(frames),
                  metadata_privacy=metadata_privacy,caption_privacy=caption_privacy,caption_relation_verified=True,
                  test_authorized=False,test_caption_values_decoded=0,raw_videos_decoded=len(frames),
                  role_assignment='source_rank_80_20_train_60_20_20_validation_seed0',
                  unknown_cross_url_visual_duplicates='not_excluded_explicit_limitation')
    result = export_pilot(out,spec,captions,selected,components,content,frames,report)
    result.update(schema_version='evidence-source-preparation-result-v1',elapsed_seconds=time.monotonic()-started,
                  source_report_sha256=file_sha256(out/'source-report.json'),test_authorized=False)
    publish_json(context.paths.provenance/'source-preparation-result.json',result)
    return result
