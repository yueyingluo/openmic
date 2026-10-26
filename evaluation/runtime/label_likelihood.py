"""Score explicit label candidates on arbitrary samples; no JSON generation or human labels."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import yaml
from common import ROOT, DESIGN, read_jsonl, sha256, unique_by, write_json
from gpu_experiment import assert_gpu_available


def candidate_suffix(prefix, full, end_id):
    if full[:len(prefix)] != prefix:
        raise ValueError('Candidate changed the prompt token prefix')
    suffix = full[len(prefix):]
    if end_id not in suffix:
        raise ValueError('Missing end-of-turn token')
    suffix = suffix[:suffix.index(end_id)+1]
    if len(suffix)<2:
        raise ValueError('Empty candidate')
    return suffix


def token_log_probabilities(logits, input_ids, prefix_length, suffix_length):
    import torch
    targets = input_ids[0,prefix_length:prefix_length+suffix_length]
    selected = torch.log_softmax(logits[0,prefix_length-1:prefix_length+suffix_length-1,:].float(),dim=-1)
    return selected.gather(1,targets[:,None]).squeeze(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','cfun'],required=True)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=900)
    args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if not cfg['authorization']['server_approved']:
        raise PermissionError('GPU execution must be approved in your local config')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not gpu or gpu!=cfg['resources']['gpu_uuid']:
        raise ValueError('An explicitly selected GPU UUID is required')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing results')
    samples=read_jsonl(args.samples)
    unique_by(samples,['sample_id'])
    if not samples or any(not isinstance(r.get('text'),str) or not r['text'].strip() for r in samples):
        raise ValueError('Each sample needs nonempty text')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    if gpu in processes.splitlines():
        raise RuntimeError('Selected GPU already has a compute process')
    # Fetching requires an explicit invocation; revision pins apply to tokenizer and weights.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one GPU required')
    torch.set_num_threads(8)
    set_seed(cfg['llm']['seed'])
    spec=cfg['llm']['models'][args.model]
    opts={'revision':spec['revision'],'trust_remote_code':False}
    if spec.get('subfolder'):
        opts['subfolder']=spec['subfolder']
    tok=AutoTokenizer.from_pretrained(spec['repo_id'],**opts)
    model=AutoModelForCausalLM.from_pretrained(spec['repo_id'],torch_dtype=torch.bfloat16,use_safetensors=True,**opts)
    assert_gpu_available()
    model=model.to('cuda').eval()
    eot=tok.convert_tokens_to_ids('<|im_end|>')
    if tok.convert_ids_to_tokens(eot)!='<|im_end|>':
        raise ValueError('Expected an explicit end-of-turn token')
    system=(DESIGN/cfg['llm']['native_cfun_recognition']['system_file']).read_text()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'metadata.json',{'model':spec,'config_sha256':sha256(args.config),
        'samples_sha256':sha256(args.samples),'source_sha256':sha256(Path(__file__)),
        'system_prompt':system,'chat_template':tok.chat_template,'torch_version':torch.__version__,
        'score':'logp(humorous+EOT)-logp(not_humorous+EOT)','length_normalization':False,
        'threshold':0,'equal_score_label':1,'calibrated_probability':False,'human_labels_consumed':False})
    started=time.monotonic()
    with (args.output/'responses.jsonl').open('x') as raw, (args.output/'predictions.jsonl').open('x') as pred:
        for row in samples:
            if time.monotonic()-started>=args.seconds:
                raise TimeoutError('Inference time budget exceeded')
            msg=[{'role':'system','content':system},{'role':'user','content':json.dumps({'text':row['text']},ensure_ascii=False)}]
            prefix=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True)
            if len(prefix)>cfg['llm']['max_input_tokens']:
                raise ValueError('Input too long; truncation forbidden')
            candidates={}
            for label in ['幽默','不幽默']:
                full=tok.apply_chat_template(msg+[{'role':'assistant','content':label}],tokenize=True,add_generation_prompt=False)
                suffix=candidate_suffix(prefix,full,eot)
                ids=torch.tensor([prefix+suffix],device='cuda')
                with torch.inference_mode():
                    output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                    values=token_log_probabilities(output.logits,ids,len(prefix),len(suffix)).cpu().tolist()
                del output
                candidates[label]={'token_ids':suffix,'token_logps':values,'logp':sum(values)}
            margin=candidates['幽默']['logp']-candidates['不幽默']['logp']
            if not math.isfinite(margin):
                raise ValueError('Non-finite model output')
            result={'sample_id':row['sample_id'],'model_id':args.model+'_label_likelihood_v1',
                'score':margin,'text_sha256':sha256(row['text'])}
            pred.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');pred.flush()
            raw.write(json.dumps(dict(result,candidates=candidates,predicted_label=int(margin>=0),
                messages_sha256=sha256(json.dumps(msg,ensure_ascii=False,sort_keys=True))),ensure_ascii=False,allow_nan=False)+'\n');raw.flush()


if __name__=='__main__':
    main()
