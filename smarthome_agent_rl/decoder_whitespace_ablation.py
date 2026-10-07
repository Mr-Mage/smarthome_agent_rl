"""A decoder-only comparison in two frozen prompt strata."""
from collections import Counter

from .review_factorial import (ARMS as FACTOR_ARMS, evaluate as factorial_evaluate,
                              parse_response as factorial_parse, request as factorial_request)

ARMS = ('original_allow', 'original_compact', 'history_allow', 'history_compact')
PROMPTS = {'original': 'plain_original', 'history': 'plain_history'}


def prompt_arm(arm):
    if arm not in ARMS:
        raise ValueError('Unknown decoder arm')
    return PROMPTS[arm.split('_', 1)[0]]


def actor_for(config, index, arm):
    prompt_arm(arm)
    offset = 2 if arm.endswith('_compact') else 0
    return config['actors'][offset + index % 2]


def request(config, item, arm):
    # Decode policy lives in the engine, so paired HTTP bodies are identical.
    return factorial_request(config, item, prompt_arm(arm))


def parse_response(context, text, arm):
    return factorial_parse(context, text, prompt_arm(arm))


def whitespace_panel(text):
    """Count whitespace outside quoted JSON strings, also for retained failures."""
    quoted = escaped = False
    count = longest = run = 0
    for character in text or '':
        if quoted:
            run = 0
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            run = 0
        elif character.isspace():
            count += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return {'characters': len(text or ''), 'outside_string_whitespace': count,
            'longest_outside_string_run': longest,
            'scope': 'Lexical scan of original text, including invalid or incomplete JSON; not a truth or token attribution metric.'}


def evaluate(config, inputs, records, historical, labels):
    forward = dict(zip(ARMS, FACTOR_ARMS))
    reverse = {value: key for key, value in forward.items()}
    if any(row['arm'] not in ARMS for row in records):
        raise ValueError('Unexpected decoder arm')
    mapped = [{**row, 'arm': forward[row['arm']]} for row in records]
    base = factorial_evaluate(config, inputs, mapped, historical['original'], labels)
    for key in ('arms', 'diagnostic_screen_passed', 'bounded_dimensions', 'binding_coverage'):
        base[key] = {reverse[arm]: value for arm, value in base[key].items()}
    base['transitions'] = {
        left + '->' + right: base['transitions'][forward[left] + '->' + forward[right]]
        for left, right in ((ARMS[0], ARMS[1]), (ARMS[2], ARMS[3]))}
    base.pop('historical_bounded_to_metadata_original')
    verdict = lambda row: row['decision']['verdict'] if row['decision'] is not None else 'INVALID'
    base['historical_transitions'] = {}
    base['whitespace'] = {}
    for prompt, source in historical.items():
        old = {row['id']: row for row in source}
        new = {row['id']: row for row in records if row['arm'] == prompt + '_allow'}
        base['historical_transitions'][prompt] = dict(Counter(
            verdict(old[item['id']]) + '->' + verdict(new[item['id']])
            if item['id'] in new else 'MISSING' for item in inputs))
    for arm in ARMS:
        rows = [row for row in records if row['arm'] == arm]
        panels = [whitespace_panel(row['call'].get('text')) for row in rows]
        base['whitespace'][arm] = {
            'records': len(rows), 'characters': sum(p['characters'] for p in panels),
            'outside_string_whitespace': sum(p['outside_string_whitespace'] for p in panels),
            'max_outside_string_run': max((p['longest_outside_string_run'] for p in panels), default=0),
            'long_run_records': sum(p['longest_outside_string_run'] >= 512 for p in panels),
            'parsed_with_outside_whitespace': sum(row['decision'] is not None and panel['outside_string_whitespace'] > 0
                                                for row, panel in zip(rows, panels)),
        }
    base['scope'] = ('Decoder-only allow/compact comparison within original and history prompt strata. '
                     'Both engines explicitly use xgrammar; only disable_any_whitespace differs. '
                     'Paired HTTP bodies and all semantic gates unchanged. Different fixed GPU replicas per decoder, '
                     'historical N88 auto-backend comparisons separate. No label repair, native admission, newSR or training.')
    return base
