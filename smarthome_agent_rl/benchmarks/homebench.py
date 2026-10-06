"""Published HomeBench instruction generation, not a device execution engine."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from ..device_contract import DeviceContract


def counters(prediction, expected):
    clean = prediction.replace(' ', '').replace('\n', '')
    generated = ','.join(re.findall(r'\{(.*?)\}', clean))
    gold = expected.replace("'''", '').replace(' ', '').replace('\n', '')
    return Counter(token for token in generated.split(',') if token), Counter(token for token in gold.split(',') if token)


def native_counts(prediction, expected):
    generated, gold = counters(prediction, expected)
    return {'exact_match': generated == gold, 'true_positive': sum((generated & gold).values()),
            'predicted': sum(generated.values()), 'expected': sum(gold.values())}


def aggregate(records):
    records = list(records)
    tp, predicted, expected = (sum(row[key] for row in records) for key in ('true_positive','predicted','expected'))
    precision, recall = tp/predicted if predicted else 0, tp/expected if expected else 0
    return {'episodes':len(records), 'exact_match':sum(row['exact_match'] for row in records)/len(records) if records else None,
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
            'counts':{'true_positive':tp,'predicted':predicted,'expected':expected},
            'zero_denominator_rule':'0; upstream compute_accuracy divides by zero for empty prediction sets'}


def qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix=qualified_name(node.value)
        return prefix+'.'+node.attr if prefix else None


class HomeBenchAdapter:
    def __init__(self, source, lock):
        self.source=Path(source)
        for name,row in lock['files'].items():
            path=self.source/name
            if not path.resolve().is_relative_to(self.source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError(f'Pinned HomeBench asset differs: {name}')
        cases=[json.loads(line) for line in (self.source/'dataset/test_data.jsonl').read_text(encoding='utf-8').splitlines()]
        homes=[json.loads(line) for line in (self.source/'dataset/home_status_method.jsonl').read_text(encoding='utf-8').splitlines()]
        self._cases={row['id']:row for row in cases}
        self._homes={row['home_id']:row for row in homes}
        if len(self._cases)!=len(cases) or len(self._homes)!=len(homes):
            raise ValueError('Duplicate task/home identity')
        if any(row['home_id'] not in self._homes for row in cases):
            raise ValueError('Published task lacks its public home')
        self.system=(self.source/'code/system.txt').read_text(encoding='utf-8')
        # Compile only this pinned, trusted upstream formatting function, never
        # the module's model-loading/main code and never generated instructions.
        tree=ast.parse((self.source/'code/model_test.py').read_text(encoding='utf-8'))
        function=next(row for row in tree.body if isinstance(row,ast.FunctionDef) and row.name=='chang_json2str')
        namespace={'__builtins__':{'str':str,'len':len}}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_homebench_formatter','exec'),namespace)
        self._format=namespace['chang_json2str']
        self._contracts={}

    def task_ids(self):
        return list(self._cases)

    def category(self,task_id):
        return self._cases[task_id]['type']

    def home_id(self,task_id):
        return self._cases[task_id]['home_id']

    def public_input(self,task_id):
        case=self._cases[task_id]
        home=self._homes[case['home_id']]
        state,methods=self._format(home['home_status'],home['method'])
        status="<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+state+'\n</home_state>\n'
        devices="<device_method>\n     The following provides the methods to control each device in the current household:"+methods+'\n</device_method>\n'
        instruction="-------------------------------\nHere are the user instructions you need to reply to.\n<User instructions:> \n"+case['input']+'\n<Machine instructions:>'
        return [{'role':'system','content':self.system+status+devices+instruction}]

    def contract(self,task_id):
        hid=self.home_id(task_id)
        if hid in self._contracts:
            return self._contracts[hid]
        home=self._homes[hid]
        spec={'capabilities':[],'functions':{}}
        shapes={}
        for method in home['method']:
            prefix=method['device_name'] if method['room_name']=='None' else method['room_name']+'.'+method['device_name']
            name=prefix+'.'+method['operation']
            spec['capabilities'].append(prefix)
            device=(home['home_status'].get(method['device_name'],{}) if method['room_name']=='None' else
                    home['home_status'].get(method['room_name'],{}).get(method['device_name'],{}))
            attributes={key.strip():value for key,value in device.get('attributes',{}).items()}
            args={}
            for parameter in method['parameters']:
                key=parameter['name']
                ptype=parameter['type']
                kind={'int':'integer','str':'string','typing.Tuple[int, int, int]':'array'}.get(ptype)
                if kind is None:
                    raise ValueError(f'Unsupported published parameter declaration: {ptype}')
                arg={'type':kind}
                if kind=='array':
                    shapes[(name,key)]=3
                attr=attributes.get(key,{})
                if kind=='integer':
                    for public,bound in [('lowest','min'),('highest','max')]:
                        if public in attr:
                            value=float(attr[public])
                            if not math.isfinite(value):
                                raise ValueError('Public attribute bound is nonfinite')
                            arg[bound]=value
                if 'options' in attr:
                    arg['enum']=attr['options']
                args[key]=arg
            if name in spec['functions']:
                raise ValueError('Duplicate public method declaration')
            spec['functions'][name]={'capability':prefix,'args':args}
        result=(DeviceContract.from_dict(f'HomeBench:{hid}',spec),shapes,spec)
        self._contracts[hid]=result
        return result

    def guard(self,task_id,prediction):
        contract,shapes,_=self.contract(task_id)
        bodies=re.findall(r'\{(.*?)\}',prediction,re.S)
        expression='['+','.join(bodies)+']'
        try:
            nodes=ast.parse(expression,mode='eval').body.elts
        except (SyntaxError,AttributeError):
            return {'prediction':prediction,'rejections':[],'uncovered':['INSTRUCTION_PARSE_UNCOVERED']}
        if not bodies:
            return {'prediction':prediction,'rejections':[],'uncovered':['INSTRUCTION_FORMAT_UNCOVERED']}
        results,rejections=[],[]
        for index,node in enumerate(nodes):
            segment=ast.get_source_segment(expression,node)
            reason=None
            if isinstance(node,ast.Name) and node.id=='error_input':
                results.append(segment)
                continue
            name=qualified_name(node.func) if isinstance(node,ast.Call) else None
            if name not in contract.functions:
                reason='UNSUPPORTED_FUNCTION'
            else:
                keys=list(contract.functions[name].arguments)
                try:
                    if len(node.args)>len(keys):
                        raise ValueError('Too many positional arguments')
                    values={key:ast.literal_eval(value) for key,value in zip(keys,node.args)}
                    for kw in node.keywords:
                        if kw.arg is None or kw.arg in values:
                            raise ValueError('Duplicate/expanded keyword argument')
                        values[kw.arg]=ast.literal_eval(kw.value)
                    for key,value in list(values.items()):
                        if (name,key) in shapes:
                            if not isinstance(value,tuple) or len(value)!=shapes[(name,key)] or any(type(v) is not int for v in value):
                                raise ValueError('Published tuple type/length mismatch')
                            values[key]=list(value)
                    validation=contract.validate({'function':name,'args':values},{})
                    if not validation.accepted:
                        reason=validation.reason_code
                except (ValueError,TypeError,SyntaxError):
                    reason='ARGUMENT_LITERAL_OR_SHAPE'
            if reason:
                rejections.append({'index':index,'instruction':segment,'reason':reason})
                results.append('error_input')
            else:
                results.append(segment)
        return {'prediction':'{'+','.join(results)+'}','rejections':rejections,'uncovered':[]}

    def score(self,task_id,prediction):
        return native_counts(prediction,self._cases[task_id]['output'])
