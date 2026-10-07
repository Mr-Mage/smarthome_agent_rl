"""Source-bound coverage graph for declared device/operation/time evidence.

All edges are lexical or interval facts, never semantic bindings. No chosen
desired phase/value, label repair, device-ID decoding, inference or dispatch.
"""
from collections import Counter
import copy
import re

from .citation_binding_report import quote_spans
from .citation_review import catalog, PERCENT, OFFSET, NUMBERS
from .effect_evidence_ablation import action_steps, checks
from .evidence_consistency import CLOCK
from .semantic_context import digest
from .semantic_verifier import VerificationContext

COMMANDS = {
    'OnOff': {'On': r'\b(?:turn|switch|power|powered)\s+on\b',
              'Off': r'\b(?:turn|switch|power|powered)\s+off\b'},
    'OperationalState': {'Start': r'\bstart\b', 'Stop': r'\bstop\b',
                         'Pause': r'\bpause\b', 'Resume': r'\bresume\b'},
}


def occurrences(text, identifier):
    if not isinstance(identifier, str) or not identifier: return []
    phrase=identifier.replace('_',' ')
    return [[m.start(),m.end()] for m in re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',text,re.I)]


def identity(context, device_id):
    """Only one complete catalog membership with a well-formed public receipt."""
    row=context['environment_state'].get('devices',{}).get(device_id)
    result={'device_id':device_id,'status':'unavailable','room':None,'device_type':None,'sources':[],
            'scope':'Observed room/type only; no numbered-device identity or anaphora resolution'}
    if not isinstance(row,dict): return result
    entries=row.get('catalog');room=row.get('room_id')
    if not isinstance(entries,list) or len(entries)!=1 or row.get('catalog_total')!=1 or row.get('catalog_truncated') is not False:
        result['status']='missing_ambiguous_or_truncated_catalog';return result
    entry=entries[0]
    if not isinstance(entry,dict):result['status']='incomplete_or_inconsistent_identity';return result
    metadata=entry.get('metadata');source=entry.get('source')
    if not isinstance(room,str) or not room or entry.get('room_id')!=room or not isinstance(metadata,dict) or not isinstance(metadata.get('device_type'),str) or not metadata['device_type']:
        result['status']='incomplete_or_inconsistent_identity';return result
    if not isinstance(source,dict) or source.get('tool')!='get_room_devices' or source.get('arguments')!={'room_id':room} or type(source.get('observation_ordinal')) is not int or source['observation_ordinal']<1 or not isinstance(source.get('response_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',source['response_sha256']):
        result['status']='invalid_catalog_receipt';return result
    result.update(status='public_catalog_words_available',room=room,device_type=metadata['device_type'],sources=[copy.deepcopy(source)])
    return result


def identity_words(text, panel):
    if panel['status']!='public_catalog_words_available':
        return {'available':False,'room_spans':[],'type_spans':[],'joint_words':False}
    room=occurrences(text,panel['room']);kind=occurrences(text,panel['device_type'])
    return {'available':True,'room_spans':room,'type_spans':kind,'joint_words':bool(room and kind)}


def operation_words(text, step):
    args=step['arguments'];family=args.get('cluster_id')
    if step['tool']=='execute_command' and family in COMMANDS and args.get('command_id') in COMMANDS[family]:
        literals=[{'span':[m.start(),m.end()],'text':m.group(0),'command_word':command}
                  for command,pattern in COMMANDS[family].items() for m in re.finditer(pattern,text,re.I)]
        literals.sort(key=lambda r:r['span'])
        return {'status':'bounded_command_vocabulary','literals':literals,'has_operation_words':bool(literals),
                'proposed_command':args['command_id'],'scope':'Words only; negation, actor, device and phase remain unresolved'}
    if step['tool']=='write_attribute' and (family,args.get('attribute_id'))==('FanControl','PercentSetting'):
        literals=[]
        for match in PERCENT.finditer(text):
            token=match.group(1).lower();value=NUMBERS[token] if token in NUMBERS else float(token)
            literals.append({'span':[match.start(),match.end()],'text':match.group(0),'value':value})
        return {'status':'bounded_percentage_vocabulary','literals':literals,'has_operation_words':bool(literals),
                'proposed_value':copy.deepcopy(args.get('value')),
                'scope':'All explicit percentage values retained; comparison does not bind device or requested phase'}
    return {'status':'uncovered_operation','literals':[],'has_operation_words':False,
            'scope':'No generic command/attribute meaning inferred from names, hidden labels or simulator'}


def time_words(text):
    offsets=[{'span':[m.start(),m.end()],'text':m.group(0),'amount':m.group(1),'unit':m.group(2),
              'anchor':'initial_public_time' if re.fullmatch(r'from\s+now',m.group(3),re.I) else 'previous_action_unbound'}
             for m in OFFSET.finditer(text)]
    clocks=[{'span':[m.start(),m.end()],'text':m.group(0),'anchor':'date_timezone_unbound'} for m in CLOCK.finditer(text)]
    return {'literals':sorted(offsets+clocks,key=lambda r:r['span']),
            'scope':'Explicit bounded time words only; no desired timestamp, prior-action binding, execution freshness or semantic correctness'}


def quote_panel(goal,quote,device,step):
    span=quote_spans(goal,quote);text=quote if span['status']=='LITERAL' else ''
    return {'literal':span,'identity':identity_words(text,device),'operation':operation_words(text,step),
            'time':time_words(text),'status':'candidate_only' if span['status']=='LITERAL' else 'unavailable'}


def interval_links(left,right):
    """Retain every occurrence pair; overlapping windows are not a phase proof."""
    return [{'left':a,'right':b,'relation':'same_span' if a==b else 'overlap' if max(a[0],b[0])<min(a[1],b[1]) else 'disjoint'}
            for a in left for b in right]


def report_record(context,row):
    VerificationContext(**context)
    before=digest({'context':context,'row':row});decision=row['decision'];checked=checks(context,decision,'combined')
    result={'original_decision_sha256':digest(decision),'raw_model_decision_sha256':digest(row['model_decision']),
            'original_verdict':decision['verdict'] if decision else 'INVALID','schema_conformant':checked['schema_conformant'],
            'literal_evidence_valid':checked['literal_evidence_valid'],'window_catalog':catalog(context['user_goal']),
            'steps':[],'semantic_status':'UNVERIFIED','dispatch_verdict':None,'task_completed':False,'native_admitted':False}
    if checked['schema_conformant']:
        goal=context['user_goal'];steps=action_steps(context)
        targets=decision['correct_target']['evidence']['steps'];claims=decision['goal_consistent']['evidence']['steps']
        for index,(step,target,claim) in enumerate(zip(steps,targets,claims),1):
            observed=identity(context,step['arguments'].get('device_id'))
            selected={name:quote_panel(goal,quote,observed,step) for name,quote in
                      [('target',target['support_quote']),('operation',claim['effect_quote']),('time',claim['requested_quote'])]}
            windows=[]
            for window in result['window_catalog']:
                text=window['text'];words=identity_words(text,observed);operation=operation_words(text,step);timing=time_words(text)
                windows.append({'citation_id':window['id'],'identity_words':words['joint_words'],
                                'operation_words':operation['has_operation_words'],'time_word_count':len(timing['literals']),
                                'time_anchors':[m['anchor'] for m in timing['literals']],
                                'operation_literal_count':len(operation['literals'])})
            links=interval_links(selected['operation']['literal']['spans'],selected['time']['literal']['spans'])
            values=selected['operation']['operation']['literals']
            flags={'selected_target_missing':selected['target']['status']=='unavailable',
                   'selected_operation_missing':selected['operation']['status']=='unavailable',
                   'selected_time_missing':selected['time']['status']=='unavailable',
                   'selected_quote_repeated':any(len(p['literal']['spans'])>1 for p in selected.values()),
                   'unsupported_with_identity_windows':target['support']=='unsupported' and any(w['identity_words'] for w in windows),
                   'selected_time_without_device_words':selected['time']['status']=='candidate_only' and observed['status']=='public_catalog_words_available' and not selected['time']['identity']['joint_words'],
                   'selected_operation_without_device_words':selected['operation']['status']=='candidate_only' and observed['status']=='public_catalog_words_available' and not selected['operation']['identity']['joint_words'],
                   'selected_operation_time_all_disjoint':bool(links) and all(l['relation']=='disjoint' for l in links),
                   'selected_previous_action_unbound':any(t['anchor']=='previous_action_unbound' for t in selected['time']['time']['literals']),
                   'selected_multi_operation_literals':len(values)>1,
                   'uncovered_operation':operation_words('',step)['status']=='uncovered_operation'}
            result['steps'].append({'step_index':index,'proposed_effect':copy.deepcopy(step),'identity':observed,
                'declared_support':target['support'],'declared_time_relation':claim['relation'],'declared_effect_relation':claim['effect_relation'],
                'selected':selected,'window_edges':windows,'selected_operation_time_links':links,'flags':flags,
                'joint_word_window_ids':[w['citation_id'] for w in windows if w['identity_words'] and w['operation_words'] and w['time_word_count']>0],
                'scope':'All lexical windows and selected interval links, no chosen identity/operation/time phase or corrected claim'})
    result['scope']='Read-only binding obligations/coverage, not entailment, numbered-device binding, reason accuracy, calibrated confidence or Task completion'
    if digest({'context':context,'row':row})!=before:raise AssertionError('Coverage report mutated source')
    return result


def evaluate(config,sources):
    arms=config['arms'];ids={s['id'] for s in sources};actual=[(s['id'],s['arm']) for s in sources]
    if arms!=['n93_joint','n93_evidence_only','n92_independent'] or len(ids)!=config['paired_inputs'] or len(sources)!=config['records'] or len(set(actual))!=len(actual) or set(actual)!={(i,a) for i in ids for a in arms}:
        raise ValueError('All source/arm/failed-record denominators required')
    records=[{k:s[k] for k in ('id','task_id','origin','arm','source_file','source_sha256','source_run')} |
             {'context_sha256':digest(s['context']),'report':report_record(s['context'],s['row'])} for s in sources]
    totals={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm];steps=[s for r in rows for s in r['report']['steps']]
        flags=Counter(k for s in steps for k,v in s['flags'].items() if v)
        totals[arm]={'records':len(rows),'schema_conformant':sum(r['report']['schema_conformant'] for r in rows),
            'literal_evidence_valid':sum(r['report']['literal_evidence_valid'] for r in rows),'steps':len(steps),
            'original_verdicts':dict(Counter(r['report']['original_verdict'] for r in rows)),
            'identity_statuses':dict(Counter(s['identity']['status'] for s in steps)),
            'operation_statuses':dict(Counter(s['selected']['operation']['operation']['status'] for s in steps)),
            'step_flags':dict(flags),'record_flags':{k:sum(any(s['flags'].get(k,False) for s in r['report']['steps']) for r in rows) for k in sorted(flags)},
            'steps_with_joint_word_windows':sum(bool(s['joint_word_window_ids']) for s in steps),
            'selected_occurrence_pair_relations':dict(Counter(l['relation'] for s in steps for l in s['selected_operation_time_links']))}
    report={'complete':True,'records':len(records),'arms':totals,
            'coverage':{'paired_inputs':len(ids),'task_count':len({s['task_id'] for s in sources}),'distinct_contexts':len({digest(s['context']) for s in sources})},
            'new_model_requests':0,'new_tokens':0,'reserved_gpu_seconds':0,'original_decisions_changed':0,'native_admitted':False,
            'semantic_status':'UNVERIFIED','scope':'Posthoc public lexical/interval coverage only. Related trajectories are not independent; missing/ambiguous/overlapping/repeated words are not semantic truth. No chosen phase,value,labels,blocking,newbenchmark orSR.'}
    return report,records
