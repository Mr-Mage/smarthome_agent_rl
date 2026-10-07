"""Declarative resource/time/state conflict detection, without arbitration."""
import copy
import hashlib
import itertools
import json
import math

LIVE_JOBS = {'REGISTERING', 'SCHEDULED', 'CLAIMED', 'UNKNOWN'}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def has_reference(value):
    if isinstance(value, str):
        return value.startswith('$')
    if isinstance(value, dict):
        return any(has_reference(child) for child in value.values())
    if isinstance(value, list):
        return any(has_reference(child) for child in value)
    return False


def resource_claims(rows, *, namespace, target_time, tolerance):
    if not isinstance(rows, (list, tuple)) or not isinstance(namespace, str) or not namespace:
        raise ValueError('Claims require a list and a nonempty resource namespace')
    result = []
    allowed = {'namespace', 'device_id', 'path', 'value', 'start_time', 'end_time', 'tolerance'}
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed or not {'device_id','path','value'} <= set(row):
            raise ValueError('Resource claim requires device_id, path and concrete target value')
        claim = copy.deepcopy(row)
        claim.setdefault('namespace', namespace)
        claim.setdefault('start_time', target_time)
        claim.setdefault('end_time', target_time + tolerance)
        claim.setdefault('tolerance', 0)
        if any(not isinstance(claim[key], str) or not claim[key] for key in ('namespace','device_id','path')):
            raise ValueError('Resource identity requires nonempty strings')
        if not all(number(claim[key]) for key in ('start_time','end_time','tolerance')):
            raise ValueError('Resource time and tolerance must be finite numbers')
        if claim['start_time'] > claim['end_time'] or claim['tolerance'] < 0:
            raise ValueError('Resource window is reversed or tolerance negative')
        if claim['tolerance'] and not number(claim['value']):
            raise ValueError('State tolerance applies only to numeric target values')
        # Round-trip permits only JSON data and fixes tuple/list representation;
        # executable expressions are never evaluated by the conflict detector.
        claim = json.loads(json.dumps(claim, allow_nan=False))
        claim['covered'] = not has_reference(claim['value'])
        result.append(claim)
    return result


def equal(left, right):
    if number(left) and number(right):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key],right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left == right


def compatible(left, right):
    if number(left['value']) and number(right['value']):
        return abs(left['value']-right['value']) <= left['tolerance']+right['tolerance']
    return equal(left['value'],right['value'])


def detect_conflicts(jobs):
    rows = [(job,index,claim) for job in jobs if job['status'] in LIVE_JOBS
            for index,claim in enumerate(job.get('resource_claims', []))]
    conflicts, uncovered = [], []
    for (left,li,a),(right,ri,b) in itertools.combinations(rows,2):
        if (a['namespace'],a['device_id'],a['path']) != (b['namespace'],b['device_id'],b['path']):
            continue
        start, end = max(a['start_time'],b['start_time']), min(a['end_time'],b['end_time'])
        if start > end:
            continue
        sides = sorted([{'job_id':left['job_id'],'task_id':left['task_id'],'task_version':left['task_version'],
                         'claim_index':li,'claim':a},
                        {'job_id':right['job_id'],'task_id':right['task_id'],'task_version':right['task_version'],
                         'claim_index':ri,'claim':b}], key=lambda row:(row['job_id'],row['claim_index']))
        evidence = {'resource':{'namespace':a['namespace'],'device_id':a['device_id'],'path':a['path']},
                    'overlap':{'start_time':start,'end_time':end},'sides':sides}
        if not a['covered'] or not b['covered']:
            uncovered.append({**evidence,'reason':'UNRESOLVED_TARGET_REFERENCE'})
            continue
        if compatible(a,b):
            continue
        id = hashlib.sha256(json.dumps(evidence,sort_keys=True,allow_nan=False).encode()).hexdigest()
        conflicts.append({'conflict_id':id,'kind':'INCOMPATIBLE_TARGET_STATES',**evidence})
    return sorted(conflicts,key=lambda row:row['conflict_id']), uncovered
