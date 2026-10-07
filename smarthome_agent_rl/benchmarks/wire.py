"""Explicit wire formats to a literal instruction sequence, without execution."""
import ast
import io
import json
import math
import tokenize

from .instructions import serialize_instructions


class ObjectPairs(list):
    """Preserve repeated method keys; a normal dict would silently drop calls."""


def compile_wire(prediction, functions, tuple_shapes):
    original=serialize_instructions(prediction)
    if not original['uncovered']:
        return {**original,'dialect':'literal_sequence'}
    text=prediction.strip()
    body=text[1:-1] if text.startswith('{') and text.endswith('}') else text
    # Presentation indentation is outside values. Multiline string arguments
    # abstain from layout recovery, rather than changing their contents.
    try:
        tokens=list(tokenize.generate_tokens(io.StringIO(body).readline))
        multiline=any(t.type==tokenize.STRING and t.start[0]!=t.end[0] for t in tokens)
        cleaned=body if multiline else '\n'.join(line.lstrip() for line in body.splitlines())
        tree=ast.parse(cleaned,mode='exec')
        pieces=[]
        for statement in tree.body:
            if not isinstance(statement,ast.Expr):
                raise ValueError('Not an instruction')
            node=statement.value
            nodes=node.elts if isinstance(node,ast.Tuple) else [node]
            for item in nodes:
                if isinstance(item,ast.Constant) and item.value=='error_input':
                    pieces.append('error_input')
                else:
                    pieces.append(ast.get_source_segment(cleaned,item))
        candidate=serialize_instructions('{'+','.join(pieces)+'}')
        if pieces and not candidate['uncovered']:
            return {**candidate,'changed':candidate['prediction']!=prediction,'dialect':'marker_or_layout'}
    except (SyntaxError,ValueError,tokenize.TokenError,IndentationError):
        pass
    # Method -> scalar or named argument object is a separately declared input
    # convention. No room/device/method alias inference occurs here.
    try:
        pairs=json.loads(text,object_pairs_hook=ObjectPairs)
    except (ValueError,TypeError):
        return {**original,'dialect':'uncovered'}
    if not isinstance(pairs,ObjectPairs) or not pairs:
        return {**original,'dialect':'uncovered'}
    calls=[]
    for name,value in pairs:
        if not isinstance(name,str) or len(name.split('.'))<2 or not all(p.isidentifier() for p in name.split('.')):
            return {**original,'dialect':'uncovered'}
        definition=functions.get(name,{})
        declarations=definition.get('args',{})
        def literal(key,data):
            if isinstance(data,ObjectPairs):
                raise ValueError('Nested argument objects are outside this public parameter dialect')
            def finite(value):
                return (not isinstance(value,float) or math.isfinite(value)) and (not isinstance(value,list) or all(finite(v) for v in value))
            if not finite(data):
                raise ValueError('Nonfinite wire value')
            if (name,key) in tuple_shapes and isinstance(data,list):
                data=tuple(data)
            return repr(data)
        try:
            if isinstance(value,ObjectPairs):
                named=dict(value)
                if len(named)!=len(value):
                    raise ValueError('Repeated argument key')
                if name in functions and set(named)==set(declarations):
                    args=[literal(k,named[k]) for k in declarations]
                else:
                    if not all(k.isidentifier() for k in named):
                        raise ValueError('Invalid argument name')
                    args=[k+'='+literal(k,v) for k,v in value]
            elif value is None and name in functions and not declarations:
                args=[]
            else:
                key=next(iter(declarations)) if len(declarations)==1 else None
                args=[literal(key,value)]
        except (ValueError,TypeError):
            return {**original,'dialect':'uncovered'}
        calls.append(name+'('+','.join(args)+')')
    candidate=serialize_instructions('{'+','.join(calls)+'}')
    return {**candidate,'changed':candidate['prediction']!=prediction,'dialect':'method_argument_map'}
