"""Four-GPU LoRA pilot with target-only loss, frozen task boundary and full receipts."""
import argparse, contextlib, hashlib, json, math, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.sft_data import tokenized_target

def main():
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader,DistributedSampler
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    config_path=ROOT/args.config;config=json.loads(config_path.read_text())
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if world!=config['world_size']:raise ValueError('Frozen four-GPU DDP required')
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    started=time.monotonic()
    random.seed(config['seed']+rank);torch.manual_seed(config['seed']+rank)
    data_path=ROOT/config['data'];audit=json.loads((ROOT/config['data_audit']).read_text())
    if not audit['admitted'] or digest(data_path)!=audit['data_sha256']:raise ValueError('Dataset admission or identity failed')
    train={r['id'] for r in json.loads((ROOT/'configs/sft-pilot/train.json').read_text())['tasks']}
    sealed={r['id'] for r in json.loads((ROOT/'configs/sft-pilot/eval.json').read_text())['tasks']}
    rows=[json.loads(line) for line in data_path.read_text().splitlines()]
    if {r['task_id'] for r in rows}-train or {r['task_id'] for r in rows}&sealed:raise ValueError('Training task leakage')
    output=ROOT/config['output']
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    write=lambda name,value:(output/name).write_text(json.dumps(value,indent=2)+'\n')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit training source first')
    import transformers,peft
    if transformers.__version__!=config['transformers']:raise ValueError('Training environment version drift')
    if rank==0:write('protocol.json',{'commit':commit,'config':config,'config_sha256':digest(config_path),
        'data_sha256':digest(data_path),'tasks':sorted({r['task_id'] for r in rows}),
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'sampler_padding_samples':math.ceil(len(rows)/world)*world-len(rows),
        'evaluation_used_for_checkpoint_selection':False})
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['base_model'],local_files_only=True)
    dataset=[tokenized_target(tokenizer,r['messages'],r['target'],config['max_length']) for r in rows]
    model=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/config['base_model'],local_files_only=True,
        dtype=torch.bfloat16,attn_implementation=config['attention']).to(local)
    model.config.use_cache=False
    targets=[name for name,module in model.named_modules() if 'language_model' in name and
             isinstance(module,torch.nn.Linear) and name.split('.')[-1] in config['lora_targets']]
    if not targets:raise ValueError('No language-only LoRA modules matched')
    model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank==0:write('model.json',{'target_modules':targets,'trainable_parameters':trainable,
        'total_parameters':sum(p.numel() for p in model.parameters()),'vision_frozen':True})
    model=DDP(model,device_ids=[local],find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,seed=config['seed'],shuffle=True)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler,collate_fn=lambda b:b[0])
    optimizer.zero_grad(set_to_none=True)
    log=(output/f'rank{rank}-steps.jsonl').open('w')
    target_count=0;sample_count=0;updates=0
    try:
        for epoch in range(config['epochs']):
            sampler.set_epoch(epoch)
            for step,row in enumerate(loader):
                group_start=(step//config['gradient_accumulation'])*config['gradient_accumulation']
                group_size=min(config['gradient_accumulation'],len(loader)-group_start)
                update=(step+1)%config['gradient_accumulation']==0 or step+1==len(loader)
                sync=contextlib.nullcontext() if update else model.no_sync()
                ids=torch.tensor([row['input_ids']],device=local)
                labels=torch.tensor(row['labels'][1:],device=local)
                positions=torch.where(labels!=-100)[0]
                targets=labels[positions]
                if not positions.numel():raise ValueError('Empty target')
                with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=positions,use_cache=False)
                    if result.logits.shape[1]!=len(targets):raise ValueError('Target/logit alignment failure')
                    loss=torch.nn.functional.cross_entropy(result.logits[0].float(),targets)
                    if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                    (loss/group_size).backward()
                if update:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
                    if not torch.isfinite(norm):raise ValueError('Non-finite gradients')
                    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                target_count+=len(targets);sample_count+=1
                log.write(json.dumps({'epoch':epoch,'step':step,'update':updates,'loss':loss.item(),
                    'input_tokens':ids.numel(),'target_tokens':len(targets),'seconds':time.monotonic()-started,
                    'memory_peak_bytes':torch.cuda.max_memory_allocated()})+'\n');log.flush()
        dist.barrier()
        if rank==0:
            model.module.save_pretrained(output/'adapter');tokenizer.save_pretrained(output/'adapter')
        dist.barrier()
        measured={'rank':rank,'samples':sample_count,'target_tokens':target_count,'optimizer_updates':updates,
            'seconds':time.monotonic()-started,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        write(f'rank{rank}-cost.json',measured)
        dist.barrier()
        if rank==0:
            receipts=[json.loads((output/f'rank{r}-cost.json').read_text()) for r in range(world)]
            write('completion.json',{'complete':True,'ranks':receipts,'allocated_gpu_seconds':sum(r['seconds'] for r in receipts),
                'adapter_sha256':{str(p.relative_to(output)):digest(p) for p in (output/'adapter').glob('*') if p.is_file()},
                'samples_including_sampler_padding':sum(r['samples'] for r in receipts)})
    except BaseException as exc:
        write(f'rank{rank}-failure.json',{'type':type(exc).__name__,'message':str(exc),'seconds':time.monotonic()-started,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'samples':sample_count})
        raise
    finally:
        log.close();dist.destroy_process_group()

if __name__=='__main__':main()
