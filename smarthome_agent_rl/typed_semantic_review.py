"""Constrain evidence syntax from the proposed action, never the expected verdict."""
from collections import Counter
import copy
import statistics

from .benchmarks.runner import valid_usage
from .semantic_diagnosis import schema as original_schema
from .semantic_prompt_ablation import evidence_check, request as prompt_request, steps


def schema(context):
    result=copy.deepcopy(original_schema())
    # The historical schema shares one decision dict across dimensions.
    # Clone each separately before specializing evidence; preserve the original.
    properties={name:copy.deepcopy(value) for name,value in result['json_schema']['schema']['properties'].items()}
    result['json_schema']['schema']['properties']=properties
    quote={'anyOf':[{'type':'string','minLength':1},{'type':'null'}]}
    for name,dimension in (('correct_target','target'),('goal_consistent','time')):
        entries=[]
        for step in steps(context['proposed_action']):
            fields={'step_index':{'type':'integer','const':step['step_index']}}
            if dimension=='target':
                fields.update({'device_id':{'const':step['device_id']},'support_quote':copy.deepcopy(quote),
                    'support':{'type':'string','enum':['requested','prerequisite','implicit','unsupported','unknown']}})
            else:
                fields.update({'encoded_execution_time':{'type':'string','const':step['encoded_execution_time']},
                    'requested_quote':copy.deepcopy(quote),
                    'relation':{'type':'string','enum':['agrees','conflict','unknown','not_applicable']}})
            entries.append({'type':'object','properties':fields,'required':list(fields),'additionalProperties':False})
        properties[name]['properties']['evidence']={'type':'object','properties':{'steps':{
            'type':'array','prefixItems':entries,'items':False,'minItems':len(entries),'maxItems':len(entries)}},
            'required':['steps'],'additionalProperties':False}
    return result


def request(config,item):
    body=prompt_request(config,item,'9b_stepwise')
    body['response_format']=schema(item['context'])
    return body


def schema_structure_check(item,row):
    if row['decision'] is None:return False
    expected=steps(item['context']['proposed_action'])
    for name,dimension in (('correct_target','target'),('goal_consistent','time')):
        evidence=row['decision'][name]['evidence']
        if set(evidence)!={'steps'} or not isinstance(evidence['steps'],list) or len(evidence['steps'])!=len(expected):return False
        for value,reference in zip(evidence['steps'],expected):
            fields={'step_index','device_id','support_quote','support'} if dimension=='target' else {
                'step_index','encoded_execution_time','requested_quote','relation'}
            if not isinstance(value,dict) or set(value)!=fields or type(value['step_index']) is not int or value['step_index']!=reference['step_index']:return False
            if dimension=='target':
                if value['device_id']!=reference['device_id'] or value['support'] not in ('requested','prerequisite','implicit','unsupported','unknown'):return False
                quote=value['support_quote']
            else:
                if value['encoded_execution_time']!=reference['encoded_execution_time'] or value['relation'] not in ('agrees','conflict','unknown','not_applicable'):return False
                quote=value['requested_quote']
            if quote is not None and (not isinstance(quote,str) or not quote):return False
    return True


def evaluate(config,inputs,records,baseline,labels):
    expected=config['proposals'];ids={r['id'] for r in inputs};categories={r['id']:r['category'] for r in labels['cases']}
    if len(inputs)!=expected or len(ids)!=expected or set(categories)!=ids or len(categories)!=len(labels['cases']):
        raise ValueError('Frozen input/developer coverage differs')
    if len(baseline)!=expected or {r['id'] for r in baseline}!=ids or any(r['arm']!='9b_stepwise' for r in baseline):
        raise ValueError('Historical N75 baseline coverage differs')
    if dict(Counter(categories.values()))!=config['developer_counts']:
        raise ValueError('Frozen developer counts differ')
    mapped={r['id']:r for r in inputs};verdict=lambda r:r['decision']['verdict'] if r['decision'] is not None else 'INVALID'
    totals={}
    for arm,rows in (('9b_historical_stepwise',baseline),('9b_typed',records)):
        evidence=[evidence_check(mapped[r['id']],{**r,'arm':'9b_stepwise'}) for r in rows]
        totals[arm]={'records':len(rows),'valid':sum(r['decision'] is not None for r in rows),
            'verdicts':dict(Counter(verdict(r) for r in rows)),
            'tokens':sum(r['call']['usage']['total_tokens'] for r in rows if valid_usage(r['call']['usage'])),
            'http_errors':sum(r['call']['error'] is not None for r in rows),
            'missing_usage':sum(not valid_usage(r['call']['usage']) for r in rows),
            'parse_errors':sum(r['parse_error'] is not None for r in rows),
            'schema_conformant':sum(schema_structure_check(mapped[r['id']],r) for r in rows),
            'median_request_seconds':statistics.median(r['call']['request_seconds'] for r in rows) if rows else None,
            'developer_conflicts':dict(Counter(verdict(r) for r in rows if categories[r['id']]=='CERTAIN_CONFLICT')),
            'developer_controls':dict(Counter(verdict(r) for r in rows if categories[r['id']]=='CONSISTENT_CONTROL')),
            'developer_unknowns':dict(Counter(verdict(r) for r in rows if categories[r['id']]=='UNCERTAIN')),
            'step_evidence':{'valid':sum(e['valid'] for e in evidence),
                'issues':dict(Counter(issue for e in evidence for issue in e['issues']))},
            'cost_origin':'immutable N75;not new cost' if arm=='9b_historical_stepwise' else 'new inference'}
    candidate=totals['9b_typed'];counts=Counter(categories.values())
    checks={'complete_records':len(records)==expected and {r['id'] for r in records}==ids and
        all(r['arm']=='9b_typed' for r in records),'http_errors':candidate['http_errors']==0,
        'usage_complete':candidate['missing_usage']==0,'structural':candidate['valid']/expected>=config['valid_ratio_min'],
        'schema_structure':candidate['schema_conformant']/expected>=config['valid_ratio_min']}
    case_gate=candidate['developer_conflicts'].get('DENY',0)==counts['CERTAIN_CONFLICT'] and candidate['developer_controls'].get('ALLOW',0)==counts['CONSISTENT_CONTROL']
    pairs=Counter();old={r['id']:r for r in baseline};new={r['id']:r for r in records}
    for identity in ids:pairs['MISSING' if identity not in new else verdict(old[identity])+'->'+verdict(new[identity])]+=1
    return {'checks':checks,'engineering_passed':all(checks.values()),'arms':totals,'pairs':dict(pairs),
        'developer_case_gate':case_gate,'diagnostic_screen_passed':all(checks.values()) and case_gate and
            candidate['step_evidence']['valid']/expected>=config['evidence_valid_ratio_min'],
        'new_model_requests':len(records),'new_model_tokens':candidate['tokens'],'native_integration_admitted':False,
        'scope':'Schema only constrains action metadata and evidence structure. Verdicts/quotes remain model output. '
                'Developer cases not independent truth;literal provenance not entailment;no native blocking,SR,Task completion or default adoption.'}
