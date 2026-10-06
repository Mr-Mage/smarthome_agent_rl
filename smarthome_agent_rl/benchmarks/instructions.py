"""Serialize literal instruction sequences; generated source is never executed."""
import ast

from .homebench import qualified_name


def serialize_instructions(prediction):
    text = prediction.strip()
    body = text[1:-1] if text.startswith('{') and text.endswith('}') else text
    try:
        tree = ast.parse(body, mode='exec')
    except (SyntaxError, ValueError):
        return {'prediction': prediction, 'changed': False, 'uncovered': ['SEQUENCE_SYNTAX_UNCOVERED']}
    nodes = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr):
            return {'prediction': prediction, 'changed': False, 'uncovered': ['NON_INSTRUCTION_STATEMENT']}
        expression = statement.value
        nodes.extend(expression.elts if isinstance(expression, ast.Tuple) else [expression])
    if not nodes:
        return {'prediction': prediction, 'changed': False, 'uncovered': ['EMPTY_INSTRUCTION_SEQUENCE']}
    segments = []
    for node in nodes:
        if isinstance(node, ast.Name) and node.id == 'error_input':
            pass
        elif isinstance(node, ast.Call) and qualified_name(node.func):
            # Symbolic argument names remain untouched for the existing Guard
            # to reject. No argument inference, quote repair or method renaming.
            def data_expression(value):
                if isinstance(value, (ast.Constant, ast.Name)):
                    return True
                if isinstance(value, (ast.Tuple, ast.List)):
                    return all(data_expression(v) for v in value.elts)
                return isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.USub, ast.UAdd)) and isinstance(value.operand, ast.Constant)
            if (any(not data_expression(a) for a in node.args) or
                    any(k.arg is None or not data_expression(k.value) for k in node.keywords)):
                return {'prediction': prediction, 'changed': False, 'uncovered': ['NON_DATA_ARGUMENT_EXPRESSION']}
        else:
            return {'prediction': prediction, 'changed': False, 'uncovered': ['NON_INSTRUCTION_EXPRESSION']}
        segments.append(ast.get_source_segment(body, node))
    serialized = '{'+','.join(segments)+'}'
    return {'prediction': serialized, 'changed': serialized != prediction, 'uncovered': []}
