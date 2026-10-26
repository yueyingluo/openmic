"""Three-seed five-class BERT; choose epochs on validation rank, never test rank."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from graded_text import metrics,read_lines,write_json,write_lines
from omni_judge import assert_idle_gpu


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,set_seed,get_linear_schedule_with_warmup
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def budget():
        if time.monotonic()-start>args.seconds:raise TimeoutError('BERT budget exceeded')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows=read_lines(args.records)
    parts={s:[] for s in ('train','validation','test')};encoded={s:[] for s in parts};excluded={s:0 for s in parts}
    for r in rows:
        item=tok(r['text'],truncation=False)
        if len(item['input_ids'])>512:
            excluded[r['split']]+=1;continue
        item['labels']=r['target']-1
        parts[r['split']].append(r);encoded[r['split']].append(item)
    collator=DataCollatorWithPadding(tok)
    def predict(model,split):
        model.eval();out=[]
        with torch.inference_mode():
            for batch in DataLoader(encoded[split],batch_size=32,collate_fn=collator):
                budget()
                inputs={k:v.to('cuda') for k,v in batch.items() if k!='labels'}
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs).logits
                out.extend((logits.float().softmax(-1)@torch.arange(1,6,device='cuda',dtype=torch.float32)).cpu().tolist())
        return np.array(out)
    all_predictions=[];summaries=[]
    for seed in [17,29,43]:
        set_seed(seed);assert_idle_gpu(args.gpu_uuid)
        model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=5,local_files_only=True).to('cuda')
        loader=DataLoader(encoded['train'],batch_size=24,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collator)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,int(len(loader)*3*.1),len(loader)*3)
        best=-2;best_epoch=None;history=[];checkpoint=args.output/f'seed-{seed}'
        for epoch in range(1,4):
            model.train()
            for batch in loader:
                budget();optimizer.zero_grad(set_to_none=True)
                inputs={k:v.to('cuda') for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs).loss
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step();scheduler.step()
            validation=metrics([r['target'] for r in parts['validation']],predict(model,'validation'))
            history.append({'epoch':epoch,'validation':validation})
            rho=validation['spearman'] if validation['spearman'] is not None else -2
            if rho>best:
                best=rho;best_epoch=epoch;model.save_pretrained(checkpoint,safe_serialization=True)
            print(json.dumps({'seed':seed,'epoch':epoch,'validation':validation}),flush=True)
        del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).to('cuda')
        predictions={s:predict(model,s) for s in ('validation','test')}
        all_predictions.append(predictions)
        summaries.append({'seed':seed,'best_epoch':best_epoch,'history':history})
        del model;gc.collect();torch.cuda.empty_cache()
    combined={s:np.mean([p[s] for p in all_predictions],axis=0) for s in ('validation','test')}
    write_lines(args.output/'predictions.private.jsonl',[{'sample_id':r['sample_id'],'split':s,'model_id':'bert_five_class_ensemble','score':float(p)}
        for s in combined for r,p in zip(parts[s],combined[s])])
    write_json(args.output/'metrics.json',{'model':'hfl/chinese-roberta-wwm-ext','revision':'5c58d0b8ec1d9014354d691c538661bf00bfdb44',
        'labels':[1,2,3,4,5],'score':'sum_k k*P(k), average three fixed seeds','epochs':3,'learning_rate':2e-5,
        'training_batch_size':24,'overlength_policy':'exclude >512 tokens, never truncate','excluded_by_split':excluded,
        'retained_by_split':{s:len(parts[s]) for s in parts},'seeds':summaries,
        'validation':metrics([r['target'] for r in parts['validation']],combined['validation']),
        'test':metrics([r['target'] for r in parts['test']],combined['test']),
        'elapsed_seconds':time.monotonic()-start,'project_human_alignment':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--seconds',type=int,default=1800)
    run(parser.parse_args())
