"""Validate an explicitly supplied listening export and unblind IDs for analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def convert(export,mapping):
    expected=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    if export.get('dataset_id')!=expected:
        raise ValueError('Ratings belong to a different listening pack')
    ids={r['blind_id']:r['sample_id'] for r in mapping}
    if len(ids)!=len(mapping):raise ValueError('Duplicate mapping IDs')
    seen=set();result=[]
    for row in export['ratings']:
        sid=row['sample_id'];rater=row['rater_id']
        if sid not in ids or sid in seen:raise ValueError('Unknown or duplicate rating ID')
        if not isinstance(rater,str) or not rater.strip():raise ValueError('Anonymous rater ID required')
        score=row['humor_score'];understood=row.get('understood')
        if score is not None and (type(score) is not int or not 1<=score<=5):
            raise ValueError('Humor score must be an integer 1-5 or null')
        if understood not in ('yes','partial','no',None):raise ValueError('Invalid comprehension value')
        if score is not None and understood not in ('yes','partial'):
            raise ValueError('Scored items require yes/partial comprehension')
        seen.add(sid)
        result.append(dict(row,sample_id=ids[sid],annotation_round='audio_pilot_v2'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=convert(json.loads(args.export.read_text()),json.loads(args.mapping.read_text()))
    with args.output.open('x') as stream:
        stream.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
