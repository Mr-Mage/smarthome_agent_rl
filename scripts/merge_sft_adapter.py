"""Create a separate inference model; never overwrite base model weights."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest

if __name__=='__main__':
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    from peft import PeftModel
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text());run=ROOT/config['output']
    completion=json.loads((run/'completion.json').read_text())
    if not completion['complete']:raise ValueError('Incomplete training')
    for name,value in completion['adapter_sha256'].items():
        if digest(run/name)!=value:raise ValueError('Adapter identity drift')
    output=run/'merged';output.mkdir(exist_ok=False)
    started=time.monotonic()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/config['base_model'],local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
    model=PeftModel.from_pretrained(model,run/'adapter').merge_and_unload(safe_merge=True)
    model.config.use_cache=True
    model.save_pretrained(output,max_shard_size='5GB',safe_serialization=True)
    AutoTokenizer.from_pretrained(ROOT/config['base_model'],local_files_only=True).save_pretrained(output)
    # Preserve the original multimodal processor identity even for text-only serving.
    import shutil
    for name in ('preprocessor_config.json','video_preprocessor_config.json'):
        source=ROOT/config['base_model']/name
        if source.exists():shutil.copy2(source,output/name)
    (run/'merge-receipt.json').write_text(json.dumps({'seconds':time.monotonic()-started,
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'files_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()},'base_overwritten':False},indent=2)+'\n')
