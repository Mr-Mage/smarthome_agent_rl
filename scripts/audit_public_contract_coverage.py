"""Audit the existing contract extractors against public source declarations.

Static candidate command names include feature-dependent commands; they are
not runtime capabilities. This audit never opens benchmark episodes or changes
rules, simulator code, model prompts, or frozen experiment thresholds.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.guard import command_contracts, public_power_rules


def cluster_id(init):
    for node in ast.walk(init):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '__init__':
            values = node.args + [kw.value for kw in node.keywords if kw.arg == 'cluster_id']
            if values and isinstance(values[0], ast.Constant) and isinstance(values[0].value, str):
                return values[0].value


def declared_commands(init, methods):
    result = set()
    # Literal dictionaries of bound methods also cover e.g. all_commands,
    # later filtered into self.commands by a feature-dependent comprehension.
    for node in ast.walk(init):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Attribute):
                if isinstance(value.value, ast.Name) and value.value.id == 'self' and value.attr in methods:
                    result.add(key.value)
    return sorted(result)


def audit(source, output):
    start = time.monotonic()
    clusters, devices = source/'src/simulator/domain/clusters', source/'src/simulator/domain/devices'
    if not clusters.is_dir() or not devices.is_dir():
        raise FileNotFoundError('Existing public SimuHome source directories required')
    output.mkdir(parents=True, exist_ok=False)
    signatures, _ = command_contracts(clusters)
    power, _ = public_power_rules(devices)
    rules = json.loads((ROOT/'configs/device-contract-rules.json').read_text(encoding='utf-8'))
    commands, device_rows, hashes = [], [], {}
    for path in sorted(clusters.glob('*.py')):
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cls in ast.parse(path.read_text(encoding='utf-8')).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            init = next((row for row in cls.body if isinstance(row, ast.FunctionDef) and row.name == '__init__'), None)
            cid = cluster_id(init) if init else None
            if cid is None:
                continue
            methods = {row.name for row in cls.body if isinstance(row, ast.FunctionDef)}
            for command in declared_commands(init, methods):
                declaration = rules.get('commands', {}).get(f'{cid}.{command}', {})
                commands.append({'cluster': cid, 'command': command, 'source': path.name,
                    'signature_extracted': (cid,command) in signatures,
                    'argument_constraints_declared': bool(declaration.get('args')),
                    'postconditions_declared': bool(declaration.get('postconditions'))})
    for path in sorted(devices.glob('*.py')):
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cls in ast.parse(path.read_text(encoding='utf-8')).body:
            if not isinstance(cls, ast.ClassDef) or path.stem in ('base','__init__'):
                continue
            methods = {row.name for row in cls.body if isinstance(row, ast.FunctionDef)}
            device_rows.append({'device_type_file': path.stem, 'class': cls.name,
                'power_pattern_extracted': path.stem in power,
                'execute_command_overridden': 'execute_command' in methods,
                'write_attribute_overridden': 'write_attribute' in methods,
                'coverage': 'partial power helper pattern only' if path.stem in power else 'no recognized device precondition pattern'})
    summary = {'device_classes': len(device_rows), 'power_patterns_extracted': sum(row['power_pattern_extracted'] for row in device_rows),
        'source_command_candidates': len(commands), 'candidate_signatures_extracted': sum(row['signature_extracted'] for row in commands),
        'candidate_postconditions_declared': sum(row['postconditions_declared'] for row in commands),
        'candidate_argument_constraints_declared': sum(row['argument_constraints_declared'] for row in commands)}
    report = {'schema':'public-source-contract-coverage-v1', 'summary':summary,
        'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()),
        'commands':commands,'devices':device_rows,'source_sha256':hashes,
        'rules_sha256':hashlib.sha256((ROOT/'configs/device-contract-rules.json').read_bytes()).hexdigest(),
        'limitations':'Static declarations, including optional/feature-dependent methods; not runtime coverage or proven complete preconditions. Attributes/enum conversion/internal state conditions not counted as covered.',
        'cost':{'seconds':time.monotonic()-start,'actor_calls':0,'judge_calls':0,'tokens':0,'allocated_gpu_seconds':0}}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'cost':report['cost'],'source_dirty':report['source_dirty']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(ROOT/'deps/SimuHome'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    audit(Path(args.source),Path(args.output))
