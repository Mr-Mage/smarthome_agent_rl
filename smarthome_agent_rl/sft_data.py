"""Task-isolated SFT splits and provenance-aware action targets; no hidden goals."""
from collections import defaultdict
import hashlib
import json
import random
import re

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def partition(rows, queries, seed=20261005):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['query_type'], row['case'])].append(row)
    if len(groups) != 12 or any(len(g) != 10 for g in groups.values()):
        raise ValueError('Expected the frozen 120-task, twelve-class development manifest')
    output = {'train':[], 'calibration':[], 'eval':[]}
    rng = random.Random(seed)
    for category, group in sorted(groups.items()):
        group = sorted(group,key=lambda r:r['id'])
        rng.shuffle(group)
        for split, selected in [('train',group[:5]),('calibration',group[5:6]),('eval',group[6:])]:
            output[split].extend(selected)
    seen_ids, seen_content, seen_queries, templates = {}, {}, {}, defaultdict(set)
    for split, tasks in output.items():
        for row in tasks:
            query = queries[row['id']].lower().strip()
            for seen, key in ((seen_ids,row['id']), (seen_content,row['sha256']), (seen_queries,text_hash(query))):
                if key in seen:
                    raise ValueError('Duplicate task/content/query across split: '+row['id'])
                seen[key]=split
            templates[text_hash(re.sub(r'\d+(?:\.\d+)?','<number>',query))].add(split)
    return output, {'numeric_normalized_cross_split_groups':sum(len(v)>1 for v in templates.values()),
                    'limitations':'Numeric query normalization is only a narrow duplicate screen; official generator template independence is not established. All 120 tasks were historically exposed; eval is training-task-isolated development diagnosis.'}

def action_target(call):
    """No labels from failed HTTP, truncated outputs, or private reasoning fields."""
    if call.get('status') != 200:
        raise ValueError('Failed HTTP')
    choice = call['response']['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Incomplete generation')
    content = choice['message'].get('content')
    body = json.loads(content)
    if set(body) != {'thought','call'} or not isinstance(body['thought'],str):
        raise ValueError('Expected the G action contract')
    action = body['call']
    if set(action) != {'tool','arguments'} or not isinstance(action['tool'],str) or not isinstance(action['arguments'],dict):
        raise ValueError('Invalid action')
    messages = call['request']['messages']
    if not messages or messages[-1]['role']=='assistant' or any(set(m)-{'role','content'} for m in messages):
        raise ValueError('Unexpected training input')
    # Retain the on-wire public conversation exactly. Score/judge/hidden goals never enter it.
    if not all(isinstance(m['content'],str) for m in messages):
        raise ValueError('Text-only inputs required')
    return messages, content, action

def tokenized_target(tokenizer, messages, target, max_length):
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    if isinstance(prefix,dict):
        prefix=prefix['input_ids']
    if prefix and isinstance(prefix[0],list):
        if len(prefix)!=1:raise ValueError('Expected a single conversation')
        prefix=prefix[0]
    # Structured inference also starts after an empty thinking block. Encode only the target plus EOT.
    suffix = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = list(prefix)+suffix
    if not all(type(token) is int for token in ids):
        raise ValueError('Tokenizer returned non-integer token IDs')
    if len(ids)>max_length:
        raise ValueError('Full target exceeds length budget; no silent truncation')
    return {'input_ids':ids, 'labels':[-100]*len(prefix)+suffix, 'attention_mask':[1]*len(ids),
            'target_tokens':len(suffix)}
