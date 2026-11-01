"""Create a local blind listening pack without exposing model scores or conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil


def build(manifest,output):
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate manifest')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audio').mkdir()
    random.Random(20261026).shuffle(rows)
    public=[];mapping=[]
    for index,row in enumerate(rows):
        source=Path(row['audio_path'])
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
        if actual!=row['audio_sha256']:raise ValueError('Audio hash mismatch')
        blind_id=f'clip-{index+1:02d}'
        shutil.copyfile(source,output/'audio'/(blind_id+'.wav'))
        public.append({'sample_id':blind_id,'audio':'audio/'+blind_id+'.wav'})
        mapping.append({'blind_id':blind_id,'sample_id':row['sample_id'],'audio_sha256':actual,
                        'topic_id':row.get('topic_id',row.get('group')),'formal_test_eligible':False})
    # Keep this identity/condition linkage outside the directory handed to a rater.
    output.with_name(output.name+'.mapping.private.json').write_text(json.dumps(mapping,indent=2))
    samples=[{'sample_id':r['sample_id'],'text':r['text'],'topic_id':r.get('topic_id',r.get('group')),
              'story_family_id':r.get('topic_id',r.get('group')),'split':'pilot','formal_test_eligible':False}
             for r in rows]
    output.with_name(output.name+'.samples.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples))
    dataset_id=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整作品听评</title>
<style>body{font:17px/1.65 system-ui;max-width:820px;margin:40px auto;padding:0 20px;color:#243243}article{border:1px solid #ccd5df;border-radius:12px;padding:22px;margin:20px 0}audio{width:100%}label{margin-right:15px}button,input,select{font:inherit;padding:7px}small{color:#526273}</style>
<h1>完整作品听评</h1><p>请戴耳机，听完整段后判断实际有多好笑。1 完全不好笑，2 略有趣，3 一般好笑，4 很好笑，5 非常好笑。听懂但不好笑可以选1；无法理解则留空，不要猜测。</p>
<p>这是开发试评，不是已经验证的正式测试。请勿查看自动评分或与其他评分者讨论后再填。</p>
<p>匿名评分者编号：<input id="rater" placeholder="例如 rater-01"></p><div id="items"></div>
<button id="save">导出我的评分 JSON</button><p id="status"></p>
<script>
const clips=__CLIPS__, datasetId=__DATASET__, key='openmic-listening-'+datasetId;
let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
document.getElementById('rater').value=state.rater||'';
const root=document.getElementById('items');
for(const c of clips){const a=document.createElement('article');a.dataset.id=c.sample_id;
 a.innerHTML='<h2>'+c.sample_id+'</h2><audio controls preload="metadata" src="'+c.audio+'"></audio><p>整体好笑程度：</p>'+[1,2,3,4,5].map(n=>'<label><input type="radio" name="'+c.sample_id+'" value="'+n+'">'+n+'</label>').join('')+' <button class="clear" type="button">清空本项分数</button><p>理解情况：<select><option value="">未填写</option><option value="yes">听懂了</option><option value="partial">部分理解</option><option value="no">无法理解</option></select></p><p>备注（可选）：<input class="note" style="width:90%"></p>';
 const old=state[c.sample_id]||{};if(old.humor_score)a.querySelector('input[value="'+old.humor_score+'"]').checked=true;
 a.querySelector('.clear').onclick=()=>{for(const input of a.querySelectorAll('input[type=radio]'))input.checked=false;collect();};
 a.querySelector('select').value=old.understood||'';a.querySelector('.note').value=old.comment||'';root.appendChild(a);}
function collect(){const rater=document.getElementById('rater').value.trim();state={rater};const ratings=[];
 for(const a of root.children){const checked=a.querySelector('input[type=radio]:checked');const row={sample_id:a.dataset.id,rater_id:rater,annotation_round:'audio_pilot_v2',humor_score:checked?Number(checked.value):null,understood:a.querySelector('select').value||null,comment:a.querySelector('.note').value};state[a.dataset.id]=row;ratings.push(row);}
 try{localStorage.setItem(key,JSON.stringify(state))}catch(e){}return ratings;}
document.addEventListener('change',collect);
document.getElementById('save').onclick=()=>{const ratings=collect();if(!state.rater){document.getElementById('status').textContent='请先填写匿名编号。';return;}
 if(ratings.some(r=>r.humor_score!==null&&!['yes','partial'].includes(r.understood))){document.getElementById('status').textContent='已打分的项目需选择听懂或部分理解；无法理解时请清空本项分数。';return;}
 const data={dataset_id:datasetId,ratings};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='listening-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('status').textContent='已导出。未填写项保持为空。';};
</script></html>'''
    body=body.replace('__CLIPS__',json.dumps(public)).replace('__DATASET__',json.dumps(dataset_id))
    (output/'index.html').write_text(body)
    (output/'README.txt').write_text('打开 index.html 听完整音频并评分，完成后导出 JSON。保留 audio 子目录。没有默认分数。\n')
    return {'clips':len(rows),'dataset_id':dataset_id,'contains_model_scores':False,'mapping_outside_pack':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.manifest,args.output)))
