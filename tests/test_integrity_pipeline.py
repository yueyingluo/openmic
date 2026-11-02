import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from openmic.tts_backends import encode_wav
from openmic.tts_stage_plan import build_plan
from openmic.tts_integrity_plan import build_integrity_plan
from openmic.tts_integrity_cli import ensure_units,assemble,IntegrityError,ASRUnavailable,parameters,dump,digest,orthography_review,content_accepted

SCRIPT='<SCRIPT>\n[SETUP]我每天都去食堂。\n[PUNCHLINE]结果只是路过食堂。\n</SCRIPT>'

class FakeBackend:
    identity={'backend':'fixture','sample_rate':16000}
    synthetic=True
    def __init__(self):self.calls=0
    def synthesize(self,**kwargs):
        self.calls+=1
        active=(np.sin(np.arange(3200)*.1)*3000).astype('<i2')
        return encode_wav(np.r_[np.zeros(1600,dtype='<i2'),active,np.zeros(1600,dtype='<i2')])

class IntegrityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.plan=build_integrity_plan(SCRIPT,build_plan(SCRIPT))
    def test_bounded_bad_content_is_never_assembled_or_re_requested_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            with patch('requests.post',side_effect=AssertionError('No network')):
                for _ in range(2):
                    with self.assertRaises(IntegrityError):
                        ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':'完全无关的错句'})
            self.assertEqual(backend.calls,4)
            self.assertFalse((out/'safe_club/audio.wav').exists())
            with self.assertRaises(IntegrityError):assemble(self.plan,[],root,out)
    def test_accepted_resume_is_hash_bound_and_does_not_re_request(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            accepted=ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':self.plan['units'][0]['text']})
            ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:(_ for _ in ()).throw(AssertionError('Already accepted')))
            self.assertEqual(backend.calls,1)
            path=root/accepted[0]['audio_path'];path.write_bytes(b'changed')
            with self.assertRaises(IntegrityError):ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{})
    def test_synthetic_test_audio_cannot_be_reported_as_real_output(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic';counter=iter(self.plan['units'])
            accepted=ensure_units(self.plan,backend,out,root,asr_function=lambda audio:{'transcript':next(counter)['text']})
            assemble(self.plan,accepted,root,out)
            self.assertTrue(json.loads((out/'safe_club/metrics.json').read_text(encoding='utf-8'))['synthetic'])

    def test_unknown_interrupted_http_attempt_is_consumed_without_resending(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic';unit=self.plan['units'][0]
            path=out/'candidates/000/01/record.json'
            dump(path,{'status':'running','params':parameters(unit,backend,1),'reference_sha256':digest(unit['text'].encode())})
            with self.assertRaises(IntegrityError):ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':'完全错误'})
            self.assertEqual(backend.calls,3)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'],'unknown_native_outcome')

    def test_asr_outage_resumes_same_downloaded_audio_without_new_tts(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            with self.assertRaises(ASRUnavailable):
                ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:(_ for _ in ()).throw(ASRUnavailable('offline service')))
            accepted=ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':self.plan['units'][0]['text']})
            self.assertEqual(backend.calls,1)
            self.assertEqual(len(accepted),1)

    def test_changed_pause_plan_is_rejected_before_reusing_accepted_audio(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':self.plan['units'][0]['text']})
            altered=json.loads(json.dumps(self.plan));altered['source_tail_pause_seconds']+=0.5
            with self.assertRaises(IntegrityError):ensure_units(altered,backend,out,root,unit_ids=[0],asr_function=lambda audio:{})
            self.assertEqual(backend.calls,1)

    def test_partial_full_file_is_archived_and_rebuilt_from_existing_pcm(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic';counter=iter(self.plan['units'])
            accepted=ensure_units(self.plan,backend,out,root,asr_function=lambda audio:{'transcript':next(counter)['text']})
            target=out/'safe_club/audio.wav';target.parent.mkdir();target.write_bytes(b'partial download')
            count=backend.calls;assemble(self.plan,accepted,root,out)
            self.assertEqual(backend.calls,count)
            self.assertTrue((target.parent/'completion.json').is_file())
            self.assertEqual(len(list((target.parent/'partial_recovery').glob('*/audio.wav'))),1)

    def test_terminal_boolean_cannot_authorize_input_changes(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            with self.assertRaisesRegex(ValueError,'boolean'):
                ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda audio:{'transcript':'错误正文'},terminal_control_approved=True)
            self.assertEqual(backend.calls,0)
            self.assertFalse((out/'candidates/000/06').exists())

    def test_special_phrase_cannot_self_authorize_without_external_manifest(self):
        review=orthography_review('我判断得没毛病。','我判断的没毛病。','我判断的没毛病。','fixture-sha')
        self.assertFalse(review['passed'])
        entry={'text':'我判断得没毛病。','content':{'passed':False},'audio_sha256':'fixture-sha','orthography_review':review}
        self.assertFalse(content_accepted(entry));self.assertFalse(entry['content']['passed'])
        for expected,transcript in [('我说不对啊。','我说不对。'),('我得去上课。','我的去上课。'),('我判断得没毛病。','我判断没毛病。')]:
            self.assertFalse(orthography_review(expected,transcript,transcript,'fixture-sha')['passed'])

    def test_verbatim_rescue_is_explicit_and_still_consumed_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);backend=FakeBackend();out=root/'topic'
            for enabled,count in ((False,4),(True,5),(True,5)):
                with self.assertRaises(IntegrityError):
                    ensure_units(self.plan,backend,out,root,unit_ids=[0],asr_function=lambda _: {'transcript':'错误正文'},verbatim_rescue=enabled)
                self.assertEqual(backend.calls,count)

if __name__=='__main__':unittest.main()
