"""Own one frozen training/merge attempt and retain wall-resource costs on failure."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config_path=ROOT/args.config;config=json.loads(config_path.read_text())
    output=ROOT/config['output']
    if output.exists():raise FileExistsError('Use a new attempt; preserve existing checkpoints and failure costs')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if len(gpu.splitlines())!=4 or any(float(v)>100 for v in gpu.splitlines()):raise ValueError('Release actor services before four-GPU training')
    python=ROOT/config['environment']/'bin/python'
    environment={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':'0,1,2,3',
        'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'TRITON_CACHE_DIR':str(ROOT/'work/sft/training-kernel-cache')}
    began=time.monotonic();commands=[];failed=None
    try:
        for argv in ([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',ROOT/'scripts/train_sft.py','--config',config_path],
                     [python,ROOT/'scripts/merge_sft_adapter.py','--config',config_path]):
            started=time.monotonic()
            result=subprocess.run(list(map(str,argv)),cwd=ROOT,env=environment)
            commands.append({'argv':list(map(str,argv)),'seconds':time.monotonic()-started,'returncode':result.returncode})
            if result.returncode:raise RuntimeError('Frozen training/merge command failed')
    except BaseException as exc:
        failed={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        output.mkdir(parents=True,exist_ok=True)
        seconds=time.monotonic()-began
        (output/'driver-receipt.json').write_text(json.dumps({'config_sha256':digest(config_path),'commands':commands,'failure':failed},indent=2)+'\n')
        (output/'resource-cost.json').write_text(json.dumps({'total_seconds_including_cleanup':seconds,
            'allocated_actor_gpu_seconds':4*seconds,'scope':'Four reserved H100 GPU wall seconds, includes training initialization/merge; not active compute or price',
            'external_judge_stopped':False,'merge_active_gpus':[0],'rank_costs_separate':True},indent=2)+'\n')
