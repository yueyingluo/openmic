import io
import hashlib
import json
import tarfile

import numpy as np
import pytest

from audio_data import safe_extract, story_splits, prepare
from graded_text import assign_groups, metrics
from graded_analysis import assemble, select, weight_fit
from omni_judge import build_messages, DIMENSIONS
from offline_project import split_speech, speech_body
from import_audio_ratings import convert
from blind_audio_pack import build


def test_duplicate_stories_never_cross_splits():
    rows=[{'canonical':'同一个很长的笑话故事','target':1},
          {'canonical':'同一个很长的笑话故事','target':5},
          {'canonical':'另外的不同故事和句子','target':3}]
    assign_groups(rows)
    assert rows[0]['group']==rows[1]['group']
    assert rows[0]['split']==rows[1]['split']


def test_constant_prediction_has_no_rank_correlation():
    result=metrics([1,2,3],[2,2,2])
    assert result['spearman'] is None
    assert result['kendall_tau_b'] is None


def test_weights_and_selection_do_not_read_test_targets(tmp_path):
    rng=np.random.default_rng(12)
    x=rng.uniform(1,5,(40,4));y=x@np.array([.6,.3,.1,0])
    w=weight_fit(x,y)
    assert w.min()>=0
    assert w.sum()==pytest.approx(1)
    assert np.mean((x@w-y)**2)<1e-6
    refs={str(i):{'target':float(y[i]),'split':'validation' if i<30 else 'test','group':str(i//2)} for i in range(40)}
    pred={name:{str(i):float(x[i,j]) for i in range(40)} for j,name in enumerate(DIMENSIONS)}
    pred.update({name:{str(i):float(x[i,0]) for i in range(40)} for name in ['direct','anchored']})
    first=select(refs,pred,'text',tmp_path/'first.json')
    for r in refs.values():
        if r['split']=='test':r['target']=999999
    second=select(refs,pred,'text',tmp_path/'second.json')
    assert first==second


def test_duplicate_predictions_rejected():
    refs=[{'sample_id':'a'}]
    row={'sample_id':'a','model_id':'m','score':3}
    with pytest.raises(ValueError,match='Duplicate'):
        assemble(refs,[row,row])


def test_reference_score_not_in_prompt_and_real_audio_present():
    row={'sample_id':'a','text':'测试','context':'语境','target':987654,'audio_path':'actual.wav'}
    prompt=build_messages(row,'audio_rubric','audio')
    assert '987654' not in json.dumps(prompt)
    assert prompt[1]['content'][0]=={'type':'audio','audio':'actual.wav'}
    assert all(c['type']!='audio' for c in build_messages(row,'text_only','audio')[1]['content'])


def test_archive_traversal_rejected(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('../escape');info.size=1
        tar.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'):
        safe_extract(archive,tmp_path/'extract')
    assert not (tmp_path/'escape').exists()


def test_speech_chunks_preserve_exact_text():
    text='第一句。'+'长句'*180+'？最后一句！'
    chunks=split_speech(text)
    assert ''.join(chunks)==text
    assert max(map(len,chunks))<=140
    assignments=story_splits([f'story-{i}' for i in range(82)])
    assert list(assignments.values()).count('test')==17
    assert list(assignments.values()).count('validation')==13
    assert assignments==story_splits(list(reversed(list(assignments))))
    assert speech_body('朗读控制指令<|endofprompt|>真正的正文')=='真正的正文'


def test_listening_import_rejects_wrong_pack_and_invented_scores():
    mapping=[{'blind_id':'clip-01','sample_id':'source-a'}]
    identity=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    row={'sample_id':'clip-01','rater_id':'anonymous-1','humor_score':None,'understood':None}
    data={'dataset_id':identity,'ratings':[row]}
    assert convert(data,mapping)[0]['humor_score'] is None
    assert convert(data,mapping)[0]['sample_id']=='source-a'
    row['humor_score']=True
    with pytest.raises(ValueError,match='integer'):
        convert(data,mapping)
    data['dataset_id']='other'
    with pytest.raises(ValueError,match='different listening pack'):
        convert(data,mapping)


def test_released_chinese_no_error_marker_is_not_a_failure(tmp_path):
    import soundfile as sf
    root=tmp_path/'corpus';root.mkdir()
    sf.write(root/'clip.wav',np.zeros(4*16000,dtype='float32'),16000)
    data={}
    for i in range(20):
        data[f'file-{i}']={'story_id':f'story-{i}','error':'无','segment_infos':{
            'sentence_segments':[{'content':'测试语句','speaker':'角色','score':[2.5],'audio_path':'clip.wav'}],
            'non_sentence_segments':[]}}
    (root/'story_infos.json').write_text(json.dumps(data,ensure_ascii=False))
    prepare(root,tmp_path/'prepared')
    summary=json.loads((tmp_path/'prepared/preparation.json').read_text())
    assert summary['eligible_rows']==20
    assert summary['stories_by_split']=={'validation':3,'test':4}


def test_blind_pack_does_not_expose_conditions_or_scores(tmp_path):
    audio=tmp_path/'source.wav';audio.write_bytes(b'fixture')
    row={'sample_id':'source-a','text':'test script','topic_id':'topic-a','condition':'hidden-source-condition',
         'model_score':4.321,'audio_path':str(audio),'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(row)+'\n')
    result=build(manifest,tmp_path/'pack')
    rendered=(tmp_path/'pack/index.html').read_text()
    assert 'hidden-source-condition' not in rendered
    assert '4.321' not in rendered
    assert 'source-a' not in rendered
    assert result['mapping_outside_pack'] is True
    assert not list((tmp_path/'pack').glob('*.json'))
