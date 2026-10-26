"""Single-GPU, bounded experiment. Explicit stage selection; no remote SSH in code."""
import argparse
import datetime
import gc
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path
import time
import numpy as np
import yaml
from common import ROOT, DESIGN, read_jsonl, sha256, write_json, write_jsonl
from judge_io import messages, validate_response

OUT = ROOT/'gpu_results'


def assert_gpu_available():
    """Recheck after potentially slow downloads; never share another compute job."""
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not allocated or not allocated.startswith('GPU-'):
        raise ValueError('One explicit GPU UUID required')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for line in processes.splitlines():
        uuid,pid=line.split(',')
        if uuid.strip()==allocated and int(pid.strip())!=os.getpid():
            raise RuntimeError('Selected GPU acquired another compute process')


def binary_metrics(y, p):
    from scipy.stats import rankdata
    y=np.asarray(y); p=np.asarray(p); pred=(p>=.5).astype(int)
    cm=np.array([[int(np.sum((y==a)&(pred==b))) for b in (0,1)] for a in (0,1)])
    f=[]
    for k in (0,1):
        tp=cm[k,k]; denom=2*tp+cm[1-k,k]+cm[k,1-k]
        f.append(float(2*tp/denom) if denom else 0.)
    pos=int(sum(y==1)); neg=int(sum(y==0))
    auc=float((rankdata(p)[y==1].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    return {'n':len(y),'macro_f1':float(np.mean(f)),
        'balanced_accuracy':float(np.mean([cm[k,k]/sum(cm[k]) for k in (0,1)])),
        'accuracy':float(np.mean(y==pred)),'roc_auc':auc,'confusion_matrix_labels_0_1':cm.tolist()}


def emit(event, **kwargs):
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kwargs}
    OUT.mkdir(exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


class Budget:
    def __init__(self,seconds): self.start=time.monotonic(); self.seconds=seconds
    def check(self):
        if time.monotonic()-self.start>=self.seconds: raise TimeoutError('Experiment wall-clock cap reached')


def load_location(name, cfg, model_root):
    if model_root:
        local=Path(model_root)/name
        if name!='bert' and cfg['llm']['models'][name].get('subfolder'):
            local=local/cfg['llm']['models'][name]['subfolder']
        if not local.exists(): raise FileNotFoundError(local)
        return str(local), {'local_files_only':True,'trust_remote_code':False}
    spec=cfg['bert'] if name=='bert' else cfg['llm']['models'][name]
    opts={'revision':spec['revision'],'token':False,'trust_remote_code':False}
    if spec.get('subfolder'): opts['subfolder']=spec['subfolder']
    return spec['repo_id'],opts


def run_bert(cfg,budget,model_root):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed
    bc=cfg['bert']; source,opts=load_location('bert',cfg,model_root)
    tok=AutoTokenizer.from_pretrained(source,**opts)
    rows=read_jsonl(ROOT/'data/recognition.jsonl')
    pilot=read_jsonl(ROOT/'data/pilot_samples.jsonl')
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','source_holdout']}
    splits['pilot']=pilot
    encoded={}
    for name,rs in splits.items():
        encoded[name]=[]
        for r in rs:
            d=tok(r['text'],truncation=False)
            if len(d['input_ids'])>bc['max_length']: raise ValueError(f'Overlength input {r["sample_id"]}')
            if 'bert_tokens' in r and len(d['input_ids'])!=r['bert_tokens']: raise ValueError('Tokenizer length differs from audited manifest')
            if 'label' in r: d['labels']=r['label']
            encoded[name].append(d)
    collator=DataCollatorWithPadding(tok)
    seed_predictions=[]; summaries=[]
    def predict(model,name):
        model.eval(); ps=[]
        loader=DataLoader(encoded[name],batch_size=bc['per_device_eval_batch_size'],collate_fn=collator)
        with torch.inference_mode():
            for batch in loader:
                budget.check()
                batch={k:v.to('cuda') for k,v in batch.items() if k!='labels'}
                with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(**batch).logits
                ps.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
        return ps
    for seed in bc['train_seeds']:
        budget.check(); set_seed(seed)
        model=AutoModelForSequenceClassification.from_pretrained(source,num_labels=2,**opts)
        assert_gpu_available()
        model=model.to('cuda')
        torch.cuda.reset_peak_memory_stats()
        loader=DataLoader(encoded['train'],batch_size=bc['per_device_train_batch_size'],shuffle=True,
            generator=torch.Generator().manual_seed(seed),collate_fn=collator)
        accum=bc['gradient_accumulation_steps']; steps=math.ceil(len(loader)/accum)*bc['epochs']
        opt=torch.optim.AdamW(model.parameters(),lr=bc['learning_rate'],weight_decay=bc['weight_decay'])
        sched=get_linear_schedule_with_warmup(opt,math.ceil(steps*bc['warmup_ratio']),steps)
        best=-1.; best_epoch=None; history=[]; checkpoint=OUT/f'bert_seed_{seed}'
        for epoch in range(1,bc['epochs']+1):
            model.train(); opt.zero_grad(set_to_none=True); loss_sum=0.
            for i,batch in enumerate(loader):
                budget.check(); batch={k:v.to('cuda') for k,v in batch.items()}
                divisor=min(accum,len(loader)-(i//accum)*accum)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(**batch).loss
                loss_sum+=float(loss.detach()); (loss/divisor).backward()
                if (i+1)%accum==0 or i+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),bc['max_grad_norm'])
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            ps=predict(model,'validation'); val=binary_metrics([r['label'] for r in splits['validation']],ps)
            history.append({'epoch':epoch,'mean_batch_loss':loss_sum/len(loader),'validation':val})
            emit('bert_epoch',seed=seed,epoch=epoch,validation_macro_f1=val['macro_f1'])
            if val['macro_f1']>best:
                best=val['macro_f1']; best_epoch=epoch
                model.save_pretrained(checkpoint,safe_serialization=True); tok.save_pretrained(checkpoint)
        peak=torch.cuda.max_memory_allocated()/1024**3
        del model,opt,sched; gc.collect(); torch.cuda.empty_cache()
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).to('cuda')
        preds={s:predict(model,s) for s in ['validation','source_holdout','pilot']}
        seed_predictions.append(preds)
        summary={'seed':seed,'best_epoch':best_epoch,'history':history,'peak_allocated_gib':peak,
                 'source_holdout':binary_metrics([r['label'] for r in splits['source_holdout']],preds['source_holdout'])}
        summaries.append(summary); write_json(OUT/f'bert_seed_{seed}_metrics.json',summary)
        write_json(OUT/f'bert_seed_{seed}_predictions.json',preds)
        del model; gc.collect(); torch.cuda.empty_cache()
    combined={s:np.mean([p[s] for p in seed_predictions],axis=0).tolist() for s in ['validation','source_holdout','pilot']}
    report={'model':'bert_binary_ensemble','model_revision':bc['revision'],
        'dataset_manifest_sha256':sha256(ROOT/'data/recognition.jsonl'),'seeds':summaries,
        'selection':'validation_macro_f1_each_seed_earliest_tie','precision':'float32_weights_bfloat16_autocast',
        'ensemble':{s:binary_metrics([r['label'] for r in splits[s]],combined[s]) for s in ['validation','source_holdout']},
        'human_alignment_validated':False}
    write_json(OUT/'bert_metrics.json',report)
    write_jsonl(OUT/'bert_pilot_predictions.jsonl',[{'sample_id':r['sample_id'],'model_id':'bert_binary_ensemble','score':p,
        'text_sha256':sha256(r['text']),'score_semantics':'source_binary_positive_probability'} for r,p in zip(pilot,combined['pilot'])])
    emit('bert_complete',source_holdout=report['ensemble']['source_holdout'])


def run_llm(name,cfg,budget,model_root):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed
    lc=cfg['llm']; source,opts=load_location(name,cfg,model_root); set_seed(lc['seed'])
    tok=AutoTokenizer.from_pretrained(source,**opts)
    model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,use_safetensors=True,**opts)
    assert_gpu_available()
    model=model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    # Start from a fresh generation config so inherited repetition/sampling settings cannot leak in.
    gen=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=lc['max_new_tokens'],use_cache=True,
                         eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id)
    write_json(OUT/f'{name}_generation.json',{'model_revision':lc['models'][name]['revision'],
        'inherited':model.generation_config.to_dict(),'resolved':gen.to_dict(),'chat_template':tok.chat_template,
        'chat_template_sha256':sha256(tok.chat_template)})
    predictions=[]; raw_path=OUT/f'{name}_responses.jsonl'
    if raw_path.exists(): raise FileExistsError('Refusing to overwrite raw model responses')
    for row in read_jsonl(ROOT/'data/pilot_samples.jsonl'):
        budget.check(); payload={'text':row['text']}; msg=messages('pointwise',payload)
        tokenized=tok.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_tensors='pt')
        if tokenized.shape[1]>lc['max_input_tokens']: raise ValueError('Input too long; no truncation allowed')
        ids=tokenized.to('cuda'); started=time.monotonic()
        with torch.inference_mode(): output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=gen)
        generated=output[0,ids.shape[1]:]; raw=tok.decode(generated,skip_special_tokens=True)
        error=None; parsed=None
        try: parsed=validate_response('pointwise',raw,payload)
        except Exception as e: error=f'{type(e).__name__}: {str(e)[:400]}'
        score=parsed['humor_score'] if parsed is not None else None
        record={'sample_id':row['sample_id'],'model_id':name,'raw_response':raw,'parsed':parsed,'validation_error':error,
            'input_tokens':ids.shape[1],'output_tokens':len(generated),'seconds':time.monotonic()-started,
            'ended_with_eos':int(generated[-1]) in ([gen.eos_token_id] if isinstance(gen.eos_token_id,int) else gen.eos_token_id),
            'messages_sha256':sha256(json.dumps(msg,ensure_ascii=False,sort_keys=True)),
            'text_sha256':sha256(row['text']),'repair_attempted':False}
        with raw_path.open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
        predictions.append({'sample_id':row['sample_id'],'model_id':name+'_pointwise','score':score,'text_sha256':sha256(row['text'])})
        emit('llm_sample',model=name,sample_id=row['sample_id'],valid=parsed is not None,output_tokens=len(generated))
    write_jsonl(OUT/f'{name}_pilot_predictions.jsonl',predictions)
    raw_rows=read_jsonl(raw_path)
    write_json(OUT/f'{name}_smoke_metrics.json',{'samples':len(predictions),'valid_json_and_schema_and_evidence':sum(r['parsed'] is not None for r in raw_rows),
        'valid_scores':sum(r['score'] is not None for r in predictions),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'repair_attempted':False,'human_alignment_validated':False,'note':'First-pass pointwise smoke test only; no prompt tuning from human scores.'})
    del model; gc.collect(); torch.cuda.empty_cache(); emit('llm_complete',model=name)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['bert','qwen','cfun'],required=True)
    p.add_argument('--model-root'); p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--config',type=Path,default=DESIGN/'experiment.yaml'); args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if list(OUT.glob(args.stage+'_*')):
        raise FileExistsError('Existing stage artifacts: use a fresh checkout/output directory; never overwrite a run')
    if not cfg['authorization']['server_approved']: raise PermissionError('No recorded user authorization')
    if not cfg['resources']['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['resources']['gpu_uuid']: raise ValueError('Explicit approved GPU UUID required')
    allocated=cfg['resources']['gpu_uuid']
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    matching=[line.strip() for line in processes.splitlines() if line.split(',')[0].strip()==allocated]
    print(json.dumps({'event':'gpu_preflight','own_pid':os.getpid(),'matching_processes':matching}),flush=True)
    if any(int(line.split(',')[1].strip())!=os.getpid() for line in matching):
        raise RuntimeError('Selected GPU acquired an existing compute process; refusing to share')
    import torch
    if torch.cuda.device_count()!=1: raise RuntimeError('Exactly one visible GPU required')
    torch.set_num_threads(8)
    OUT.mkdir(exist_ok=True)
    versions={x:importlib.metadata.version(x) for x in ['torch','transformers','numpy','scipy','jsonschema','PyYAML']}
    write_json(OUT/f'{args.stage}_environment.json',{'versions':versions,'cuda_runtime':torch.version.cuda,
        'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'device_name':torch.cuda.get_device_name(0),
        'config_sha256':sha256(args.config),'config_path':str(args.config.resolve()),
        'stage':args.stage,'wall_cap_seconds':args.seconds,
        'artifact_sha256':{str(path.relative_to(DESIGN)):sha256(path) for path in
            [Path(__file__).resolve(),ROOT/'common.py',ROOT/'judge_io.py',ROOT/'data/pilot_samples.jsonl',
             DESIGN/'prompts/pointwise_system.txt',DESIGN/'prompts/pointwise_user.txt',
             DESIGN/'schemas/pointwise_response.schema.json']}})
    budget=Budget(args.seconds); emit('stage_start',stage=args.stage)
    try:
        if args.stage=='bert': run_bert(cfg,budget,args.model_root)
        else: run_llm(args.stage,cfg,budget,args.model_root)
    except Exception as exc:
        emit('stage_failed',stage=args.stage,error=type(exc).__name__,detail=str(exc)[:500]); raise
    emit('stage_complete',stage=args.stage,elapsed_seconds=time.monotonic()-budget.start)


if __name__=='__main__': main()
