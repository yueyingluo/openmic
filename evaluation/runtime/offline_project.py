"""Real OpenMic AutoGen workflow against a loopback local model, plus offline speech.

This is an explicitly different inference/TTS configuration from the production
API setup. Generated works are technical pilots, never human-rated test data.
"""
import argparse
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

from omni_judge import assert_idle_gpu

SYSTEM = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
TOPICS = ['宿舍里每个人都设闹钟，最后只有不需要早起的人被叫醒', '食堂排队时不断换到看起来更快的队伍']


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def split_speech(text,limit=140):
    chunks=[]
    current=''
    for piece in re.findall(r'[^。！？!?；;\n]+[。！？!?；;\n]*',text):
        if len(current)+len(piece)>limit and current:
            chunks.append(current);current=''
        while len(piece)>limit:
            chunks.append(piece[:limit]);piece=piece[limit:]
        current+=piece
    if current:chunks.append(current)
    if ''.join(chunks)!=text:
        raise ValueError('Speech split lost characters')
    return chunks


def speech_body(prepared_text):
    """CosyVoice's instruction prefix must not be spoken by an Omni renderer."""
    return prepared_text.split('<|endofprompt|>',1)[-1].strip()


def generate(args):
    # This stage is entirely offline except its own loopback HTTP endpoint.
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key,None)
    import torch
    from transformers import Qwen2_5OmniProcessor,Qwen2_5OmniThinkerForConditionalGeneration
    from openmic.autogen_v02 import AutoGenConfig,AutoGenV02Workflow
    from openmic.models import ProjectRequest
    from openmic.tts import extract_spoken_script,prepare_tts_text
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'127.0.0.1','localhost']))
    assert_idle_gpu(args.gpu_uuid)
    processor=Qwen2_5OmniProcessor.from_pretrained(args.model,local_files_only=True)
    model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
    token=secrets.token_hex(24)
    calls=[]
    def completion(messages):
        normalized=[]
        for m in messages:
            role=m['role'] if m['role'] in ('user','assistant','system') else 'user'
            content=m.get('content','') or ''
            if not isinstance(content,str):raise ValueError('Only text chat supported')
            normalized.append({'role':role,'content':[{'type':'text','text':content}]})
        prompt=processor.apply_chat_template(normalized,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=prompt,return_tensors='pt').to(model.device)
        before=time.monotonic()
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=2048,do_sample=False,
                               eos_token_id=processor.tokenizer.eos_token_id,
                               pad_token_id=processor.tokenizer.pad_token_id)
        new=ids[0,inputs['input_ids'].shape[1]:]
        text=processor.tokenizer.decode(new,skip_special_tokens=True)
        calls.append({'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':inputs['input_ids'].shape[1],
                      'output_tokens':len(new),'elapsed_seconds':time.monotonic()-before,
                      'truncated':int(new[-1])!=processor.tokenizer.eos_token_id})
        if calls[-1]['truncated']:
            raise RuntimeError('Generation hit its token cap without end-of-turn; not a complete script')
        return text,inputs['input_ids'].shape[1],len(new)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(403);return
            try:
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text,nin,nout=completion(payload['messages'])
                data=json.dumps({'id':'local-'+str(len(calls)),'object':'chat.completion','created':int(time.time()),'model':'omni-local',
                    'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'length' if nout>=2048 else 'stop'}],
                    'usage':{'prompt_tokens':nin,'completion_tokens':nout,'total_tokens':nin+nout}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except Exception as exc:
                self.send_error(500,type(exc).__name__)
        def log_message(self,*_):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    scripts=[]
    try:
        for index,topic in enumerate(TOPICS[:args.topics]):
            request=ProjectRequest(topic=topic,duration_minutes=3)
            text,_,_=completion([{'role':'system','content':'你是中文单口喜剧编剧。'},
                                {'role':'user','content':f'围绕“{topic}”写面向大学生的3分钟完整单口稿，约800到1000汉字，有铺垫、笑点和回调，只输出正文。'}])
            scripts.append({'script_id':f'topic{index}-single','topic_id':f'topic{index}','condition':'single_agent_local',
                            'request':asdict(request),'text':text,'spoken_text':speech_body(prepare_tts_text(text)),'messages':[]})
            write(args.output/'scripts.private.json',scripts)
            config=AutoGenConfig(api_key=token,base_url=f'http://127.0.0.1:{server.server_port}/v1',model='omni-local',temperature=0,timeout_seconds=600)
            result=AutoGenV02Workflow(config,max_revisions=0).run(request)
            coach=[m.content for m in result.messages if m.agent=='PerformanceCoach']
            if not coach or '<SCRIPT>' not in coach[-1] or '</SCRIPT>' not in coach[-1]:
                raise ValueError('Workflow did not produce a complete performance SCRIPT block')
            text=extract_spoken_script(coach[-1])
            scripts.append({'script_id':f'topic{index}-multi','topic_id':f'topic{index}','condition':'five_agent_local',
                            'request':asdict(request),'text':text,'spoken_text':speech_body(prepare_tts_text(text)),'workflow':asdict(result)})
            write(args.output/'scripts.private.json',scripts)
    finally:
        server.shutdown();server.server_close();worker.join()
        write(args.output/'generation_calls.json',calls)
    write(args.output/'generation_manifest.json',{'project_revision':args.project_revision,'model_revision':args.revision,
        'real_autogen_workflow':True,'max_revisions':0,'sampling':'greedy','topics':TOPICS[:args.topics],
        'scripts':len(scripts),'external_api_calls':0,'production_configuration':False,'human_validation':False,
        'eos_token_id':processor.tokenizer.eos_token_id,'all_generations_end_at_eos':all(not c['truncated'] for c in calls)})


def speak(args):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Qwen2_5OmniProcessor,Qwen2_5OmniForConditionalGeneration,LogitsProcessor,LogitsProcessorList
    scripts=json.loads((args.output/'scripts.private.json').read_text())
    audio_root=args.output/'audio';audio_root.mkdir(exist_ok=False)
    assert_idle_gpu(args.gpu_uuid)
    processor=Qwen2_5OmniProcessor.from_pretrained(args.model,local_files_only=True)
    model=Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
    class ForcedTranscript(LogitsProcessor):
        def __init__(self,tokens,prefix):self.tokens=tokens;self.prefix=prefix
        def __call__(self,input_ids,scores):
            step=input_ids.shape[1]-self.prefix
            scores.fill_(float('-inf'));scores[:,self.tokens[min(step,len(self.tokens)-1)]]=0
            return scores
    records=[]
    for script in scripts:
        chunks=split_speech(script['spoken_text'])
        pieces=[];segments=[];cursor=0.0
        for index,chunk in enumerate(chunks):
            torch.manual_seed(20261026+index)
            messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                      {'role':'user','content':[{'type':'text','text':'请逐字朗读以下中文，不加开场或解释：'+chunk}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=prompt,return_tensors='pt').to(model.device)
            tokens=processor.tokenizer.encode(chunk,add_special_tokens=False)+[processor.tokenizer.eos_token_id]
            with torch.inference_mode():
                ids,wave=model.generate(**inputs,return_audio=True,speaker='Chelsie',
                    thinker_max_new_tokens=len(tokens),thinker_do_sample=False,
                    thinker_eos_token_id=processor.tokenizer.eos_token_id,
                    thinker_pad_token_id=processor.tokenizer.pad_token_id,
                    thinker_logits_processor=LogitsProcessorList([ForcedTranscript(tokens,inputs['input_ids'].shape[1])]),
                    talker_max_new_tokens=4096)
            actual=processor.tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            if actual!=chunk:raise ValueError('Forced text transcript mismatch')
            wave=wave.detach().float().cpu().numpy().reshape(-1)
            if not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid synthesized audio')
            segment_path=audio_root/f'{script["script_id"]}-{index:02d}.wav'
            sf.write(segment_path,wave,24000,subtype='PCM_16')
            duration=len(wave)/24000
            segments.append({'index':index,'text':chunk,'start':cursor,'end':cursor+duration,'path':str(segment_path)})
            pieces.extend([wave,np.zeros(6000,dtype='float32')]);cursor+=duration+.25
            print(json.dumps({'script':script['script_id'],'speech_chunk':index+1,'total':len(chunks)}),flush=True)
        combined=np.concatenate(pieces[:-1])
        path=audio_root/(script['script_id']+'.wav');sf.write(path,combined,24000,subtype='PCM_16')
        row={'sample_id':script['script_id'],'script_id':script['script_id'],'topic_id':script['topic_id'],'group':script['topic_id'],
             'split':'pilot','text':script['spoken_text'],'audio_path':str(path),'duration_seconds':len(combined)/24000,
             'sample_rate':24000,'channels':1,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'text_sha256':hashlib.sha256(script['spoken_text'].encode()).hexdigest(),'condition':script['condition'],
             'tts':'Qwen2.5-Omni-7B forced transcript, Chelsie, <=140-character chunks + 250ms boundaries',
             'asr_or_human_transcript_verified':False,'formal_test_eligible':False,'segments':segments}
        records.append(row)
        (args.output/'audio_manifest.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write(args.output/'speech_completed.json',{'audio_files':len(records),'model_revision':args.revision,'production_cosyvoice':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['generate','speak'],required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--project-revision',default='unknown')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--topics',type=int,default=2,choices=[1,2])
    args=parser.parse_args()
    (generate if args.stage=='generate' else speak)(args)
