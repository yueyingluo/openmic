"""Prompt assembly and response validation only; never calls a model or endpoint."""
import json
import yaml
from jsonschema import Draft202012Validator
from common import DESIGN, ROOT, sha256, read_json, write_json, write_jsonl


def messages(task, payload):
    if task not in ('pointwise','pairwise'): raise ValueError('Unknown task')
    keys={'text'} if task=='pointwise' else {'A','B'}
    if set(payload)!=keys or any(not isinstance(v,str) or not v.strip() for v in payload.values()):
        raise ValueError('Invalid input object')
    system=(DESIGN/f'prompts/{task}_system.txt').read_text()
    template=(DESIGN/f'prompts/{task}_user.txt').read_text()
    if template.count('{{INPUT_JSON}}')!=1: raise ValueError('Expected one placeholder')
    return [{'role':'system','content':system},{'role':'user','content':template.replace('{{INPUT_JSON}}',json.dumps(payload,ensure_ascii=False))}]


def validate_response(task, raw, payload):
    def no_constant(v): raise ValueError(f'Nonstandard JSON constant {v}')
    def unique_object(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError(f'Duplicate JSON key {k}')
            out[k]=v
        return out
    obj=json.loads(raw,parse_constant=no_constant,object_pairs_hook=unique_object)
    schema=read_json(DESIGN/f'schemas/{task}_response.schema.json')
    Draft202012Validator(schema).validate(obj)
    evidence=[('evidence','text')] if task=='pointwise' else [('evidence_a','A'),('evidence_b','B')]
    for field,key in evidence:
        if any(q not in payload[key] for q in obj[field]): raise ValueError('Evidence is not a verbatim substring')
    return obj


def reconcile_pair(ab, ba):
    if ab['status']!='judged' or ba['status']!='judged': return 'invalid_or_abstained'
    mapped={'A':'B','B':'A','tie':'tie'}[ba['winner']]
    return ab['winner'] if ab['winner']==mapped else 'order_inconsistent'


def main():
    config=yaml.safe_load((DESIGN/'experiment.yaml').read_text())
    for task in ['pointwise','pairwise']:
        Draft202012Validator.check_schema(read_json(DESIGN/f'schemas/{task}_response.schema.json'))
    files=[DESIGN/'experiment.yaml']+sorted((DESIGN/'prompts').glob('*.txt'))+sorted((DESIGN/'schemas').glob('*.json'))
    write_json(ROOT/'reports/config.lock.json',{'scope':'local_preflight_only_not_runtime_environment_lock',
        'authorization':config['authorization'],'files':{str(p.relative_to(DESIGN)):sha256(p) for p in files},
        'gpu_or_api_calls':0,'model_weights_downloaded':False})
    pilot=read_json(DESIGN/'pilot_manifest.json')
    requests=[]
    samples=[]
    for s in pilot['samples']:
        samples.append(dict(sample_id=s['sample_id'],text=s['text'],topic_id=s['theme'],
                            story_family_id=s['sample_id'],split='pilot',formal_test_eligible=False))
        msg=messages('pointwise',{'text':s['text']})
        for name,spec in config['llm']['models'].items():
            requests.append({'sample_id':s['sample_id'],'model_id':name,'model':spec,'task':'pointwise',
                'text_sha256':sha256(s['text']),'messages':msg,'messages_sha256':sha256(json.dumps(msg,ensure_ascii=False,sort_keys=True)),
                'state':'prepared_not_executed','source':'assistant_authored_pilot','formal_test_eligible':False})
    write_jsonl(ROOT/'data/pilot_samples.jsonl',samples)
    write_jsonl(ROOT/'data/pilot_requests.jsonl',requests)
    print(f'Validated schemas; prepared {len(requests)} requests; model calls = 0')


if __name__=='__main__': main()
