"""Prepare human-scored CEAEval audio without treating weak labels as human data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive, output):
    output.mkdir(parents=True,exist_ok=False)
    root=output.resolve()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            dest=(root/member.name).resolve()
            if dest!=root and root not in dest.parents:
                raise ValueError('Unsafe archive path')
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError('Unsupported archive entry')
            if member.isdir():
                dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source, dest.open('xb') as target:
                    while True:
                        chunk=source.read(8*1024*1024)
                        if not chunk:break
                        target.write(chunk)


def story_splits(stories):
    """Fixed hash ordering with explicit counts prevents a tiny development split."""
    ordered=sorted(set(stories),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    test_count=math.ceil(.2*len(ordered));validation_count=math.ceil(.15*len(ordered))
    return {s:('test' if i<test_count else 'validation' if i<test_count+validation_count else 'train')
            for i,s in enumerate(ordered)}


def prepare(root, output):
    import numpy as np
    import soundfile as sf
    output.mkdir(parents=True,exist_ok=False)
    files=list(root.rglob('story_infos.json'))
    if len(files)!=1:
        raise ValueError('Expected exactly one story_infos.json')
    metadata=json.loads(files[0].read_text())
    base=files[0].parent
    assignments=story_splits([entry['story_id'] for entry in metadata.values()])
    records=[]
    exclusions=Counter()
    for file_id,entry in sorted(metadata.items()):
        # Released JSON uses the Chinese string "无" for no source error.
        if entry.get('error') not in (None, False, '', '无', 'none', 'None'):
            exclusions['source_error_files']+=1
            continue
        story=entry['story_id']
        previous=[]
        for i,segment in enumerate(entry['segment_infos']['sentence_segments']):
            text=segment.get('content','').strip()
            context='\n'.join(previous)[-2000:]
            previous.append(str(segment.get('speaker',''))+'：'+text)
            values=segment.get('score',[])
            if isinstance(values,(int,float)):
                values=[values]
            if not values or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=5 for v in values):
                exclusions['missing_or_invalid_score']+=1
                continue
            path=(base/segment['audio_path']).resolve()
            if base.resolve() not in path.parents or not path.is_file():
                raise ValueError('Missing or out-of-root audio')
            info=sf.info(path)
            if not text or not 3<=info.duration<=20:
                exclusions['empty_text_or_outside_3_20_seconds']+=1
                continue
            sample_id=hashlib.sha256((str(file_id)+'/'+str(i)+'/'+segment['audio_path']).encode()).hexdigest()[:20]
            records.append({'sample_id':sample_id,'story_id':story,'group':story,'split':assignments[story],
                'text':text,'context':context,'audio_path':str(path),'duration_seconds':info.duration,
                'sample_rate':info.samplerate,'channels':info.channels,'target':sum(values)/len(values),
                'n_reference_scores':len(values)})
    if len({r['sample_id'] for r in records})!=len(records):
        raise ValueError('Duplicate audio identity')
    chosen=[]
    for split,cap in [('validation',64),('test',128)]:
        counts=Counter()
        pool=sorted((r for r in records if r['split']==split),key=lambda r:hashlib.sha256(('audio-v2-'+r['sample_id']).encode()).hexdigest())
        for row in pool:
            if counts[row['group']]>=6:
                continue
            counts[row['group']]+=1
            chosen.append(row)
            if sum(counts.values())>=cap:break
    if any(not any(r['split']==s for r in chosen) for s in ('validation','test')):
        raise ValueError('No eligible validation/test audio; inspect source schema')
    for r in chosen:
        data,_=sf.read(r['audio_path'],dtype='float32',always_2d=True)
        if not np.isfinite(data).all() or not len(data):
            raise ValueError('Nonfinite or empty waveform')
        r['audio_sha256']=digest(Path(r['audio_path']))
    for name,rows in [('references.private.jsonl',chosen),('judge_inputs.jsonl',[{k:v for k,v in r.items() if k not in ('target','n_reference_scores')} for r in chosen])]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'dataset':'CEAEval-D manual_annotation','revision':'f737f27f21f073e7a41bf52b0d4ffa40db9f8b08',
            'metadata_files':len(metadata),'unique_story_ids':len({v['story_id'] for v in metadata.values()}),
            'raw_sentence_segments':sum(len(v['segment_infos']['sentence_segments']) for v in metadata.values()),
            'raw_non_sentence_segments':sum(len(v['segment_infos']['non_sentence_segments']) for v in metadata.values()),
            'eligible_rows':len(records),'exclusions':dict(exclusions),
            'selected_rows':len(chosen),'split_counts':dict(Counter(r['split'] for r in chosen)),
            'stories_by_split':{s:len({r['group'] for r in chosen if r['split']==s}) for s in ('validation','test')},
            'score_range':[0,5],'actual_reference_score_counts':dict(Counter(r['n_reference_scores'] for r in chosen)),
            'selection':'sha256 identity order, max 6 clips/story, duration 3-20s, no score balancing',
            'story_split_rule':'sort unique story IDs by sha256; first ceil(20%) test, next ceil(15%) validation, rest train',
            'context':'preceding utterances within source file, last 2000 chars; not complete novel context',
            'target_mean_by_split':{s:float(np.mean([r['target'] for r in chosen if r['split']==s])) for s in ('validation','test')},
            'metadata_sha256':digest(files[0]),'input_sha256':digest(output/'judge_inputs.jsonl'),
            'project_humor_alignment':False}
    (output/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--archive-sha256')
    args=parser.parse_args()
    if args.archive:
        if not args.archive_sha256 or digest(args.archive)!=args.archive_sha256:
            raise ValueError('Archive hash required and must match')
        safe_extract(args.archive,args.root)
    prepare(args.root,args.output)
