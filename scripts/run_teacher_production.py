"""Produce real teacher episodes on the external A800; reserve no local actor GPUs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest
import httpx

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True);args=p.parse_args()
    config_path=ROOT/args.config;config=json.loads(config_path.read_text())
    run=ROOT/args.run_dir;run.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Freeze teacher source first')
    receipt={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config_sha256':digest(config_path),'config':config,'local_actor_gpus_reserved':0,
        'teacher_judge_shared_model':True,'phase':'training-data production, not model-benefit evaluation'}
    (run/'protocol.json').write_text(json.dumps(receipt,indent=2)+'\n')
    retrieval=json.loads((ROOT/'configs/p1-qwen35-9b-retrieval.json').read_text())['retrieval']
    start=time.monotonic();process=None
    with (run/'embedding.log').open('w') as output:
      try:
        with httpx.Client(trust_env=False,timeout=10) as client:
            response=client.get(config['judge_endpoint']+'/models');response.raise_for_status()
            (run/'external-models.json').write_text(json.dumps(response.json(),indent=2)+'\n')
            if not any(r['id']==config['actor_model'] for r in response.json()['data']):raise ValueError('Teacher model unavailable')
        process=subprocess.Popen([retrieval['python'],ROOT/'scripts/serve_doc_embeddings.py','--model',retrieval['model_path'],
            '--port',str(config['embedding_port'])],cwd=ROOT,env={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':''},
            stdout=output,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Embedding service exited')
            try:
                with httpx.Client(trust_env=False,timeout=2) as client:
                    if client.get(f"http://127.0.0.1:{config['embedding_port']}/health").is_success:break
            except httpx.TransportError:pass
            time.sleep(1)
        else:raise TimeoutError('Embedding startup')
        for script,values in [('run_benchmark_suite.py',['--phase','dev','--variants','Teacher','--manifest','configs/sft-pilot/train.json',
             '--config',config_path,'--run-dir',run/'episodes']),('report_benchmark.py',[run/'episodes']),('verify_benchmark.py',[run/'episodes'])]:
            subprocess.run([sys.executable,ROOT/'scripts'/script,*map(str,values)],cwd=ROOT,check=True)
        (run/'completion.json').write_text(json.dumps({'complete':True,'teacher_episodes':120})+'\n')
      except BaseException as exc:
        (run/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)})+'\n');raise
      finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
        (run/'resource-cost.json').write_text(json.dumps({'total_seconds_including_cleanup':time.monotonic()-start,
            'allocated_actor_gpu_seconds':0,'external_judge_stopped':False,
            'scope':'External teacher and judge on shared A800, GPU allocation unmeasured; no local actor GPUs reserved by this attempt'},indent=2)+'\n')
