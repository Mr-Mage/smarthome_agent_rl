"""Typed public-action review with failure isolation and immutable HTTP receipts.

This adapter is report-only. A valid DENY, like a failed review, never changes
the actor's observations or the original tool dispatch.
"""
import copy
from dataclasses import asdict
import math
import time

from .benchmarks.runner import completion, valid_usage
from .semantic_context import digest
from .semantic_diagnosis import parse
from .semantic_prompt_ablation import evidence_check
from .semantic_verifier import Decision, VerificationContext, VerificationResult
from .typed_semantic_review import request, schema_structure_check

NAMES = ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute')


def uncertain(reason):
    return VerificationResult(*(Decision('UNCERTAIN', 0.0, {'review_failure': reason})
                                for _ in NAMES), reason_code=reason)


def result_from(value):
    return VerificationResult(*(Decision(**value[name]) for name in NAMES),
                              reason_code=value['reason_code'])


class ReportSemanticReviewer:
    report_only = True

    def __init__(self, config, endpoint, *, transport=completion, audit_fn=None):
        self.config = copy.deepcopy(config)
        timeout = self.config['request_timeout']
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Review timeout must be a finite positive number')
        self.endpoint, self.transport, self.audit_fn = endpoint, transport, audit_fn
        self.records = []

    def verify(self, context: VerificationContext) -> VerificationResult:
        # Recheck even an existing frozen dataclass: nested mappings are mutable.
        # Policy/source leakage is a configuration error, never an HTTP fallback.
        public = asdict(context)
        VerificationContext(**public)
        item = {'context': public}
        body = request(self.config, item)
        started = time.monotonic()
        try:
            call = self.transport(self.endpoint, copy.deepcopy(body), self.config['request_timeout'])
        except Exception as exc:
            call = {'endpoint': self.endpoint, 'body': body, 'response': None,
                    'error': {'type': type(exc).__name__, 'message': str(exc)},
                    'text': '', 'usage': None, 'finish_reason': None,
                    'request_seconds': time.monotonic() - started}
        decision, parse_error, schema_ok, evidence = None, None, False, None
        reason = None
        try:
            if call.get('body') != body or call.get('endpoint') != self.endpoint:
                reason = 'REVIEW_RECEIPT_MISMATCH'
            elif call.get('error') is not None:
                reason = 'REVIEW_TRANSPORT_FAILURE'
            else:
                try:
                    decision = parse(call['text'])
                except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
                    parse_error = {'type': type(exc).__name__, 'message': str(exc)}
                if decision is not None:
                    row = {'decision': decision, 'arm': '9b_stepwise'}
                    schema_ok = schema_structure_check(item, row)
                    evidence = evidence_check(item, row)
                if call.get('finish_reason') != 'stop':
                    reason = 'REVIEW_INCOMPLETE_OUTPUT'
                elif decision is None:
                    reason = 'REVIEW_PARSE_FAILURE'
                elif not schema_ok:
                    reason = 'REVIEW_SCHEMA_FAILURE'
                elif not evidence['valid']:
                    reason = 'REVIEW_EVIDENCE_FAILURE'
                elif not valid_usage(call.get('usage')):
                    reason = 'REVIEW_USAGE_MISSING'
                elif call.get('response', {}).get('model') != body['model']:
                    reason = 'REVIEW_MODEL_MISMATCH'
        except Exception as exc:
            # Unexpected provider payloads also cannot abort an admitted action.
            reason = 'REVIEW_PAYLOAD_FAILURE'
            parse_error = {'type': type(exc).__name__, 'message': str(exc)}
        result = uncertain(reason) if reason else result_from(decision)
        self.records.append(copy.deepcopy({
            'sequence': len(self.records) + 1, 'context': public,
            'context_sha256': digest(public), 'request_sha256': digest(body),
            'call': call, 'model_decision': decision, 'parse_error': parse_error,
            'schema_conformant': schema_ok, 'evidence': evidence,
            'fallback_reason': reason, 'accepted_model_output': reason is None,
            'result': result.as_dict(),
        }))
        # Evidence persistence errors remain infrastructure failures, not hidden
        # successes. Transport/output failures above are already fully isolated.
        if self.audit_fn:
            self.audit_fn(self.records)
        return result

    def costs(self):
        calls = [r['call'] for r in self.records]
        return {'requests': len(calls),
                'tokens': sum(c['usage']['total_tokens'] for c in calls if valid_usage(c.get('usage'))),
                'missing_usage': sum(not valid_usage(c.get('usage')) for c in calls),
                'request_seconds': sum(c.get('request_seconds', 0) for c in calls),
                'accepted_model_outputs': sum(r['accepted_model_output'] for r in self.records),
                'fallbacks': sum(r['fallback_reason'] is not None for r in self.records)}


def attach_report_reviewer(config, variant, *, audit_fn=None, transport=completion):
    """Resolve an explicit serializable report policy; default G is untouched."""
    policy = copy.deepcopy(config.get('variant_policies', {}).get(variant))
    if not policy or not policy.get('report_semantic_review'):
        return policy, None
    additions = {'report_semantic_review', 'semantic_context_version',
                 'semantic_context_references', 'semantic_context_workflow'}
    if ({k: v for k, v in policy.items() if k not in additions} !=
            config['variant_policies'].get('G')):
        raise ValueError('Report review must preserve the frozen G execution policy')
    if any(policy.get(k) for k in ('semantic_blocking', 'semantic_verifier', 'reflection_verifier',
                                  'task_runtime', 'execution_runtime')):
        raise ValueError('Report review forbids blocking, runtime and verifier combinations')
    if policy.get('semantic_context_version') != 2 or not all(policy.get(k) is True for k in
            ('report_semantic_review', 'semantic_context_references', 'semantic_context_workflow')):
        raise ValueError('Report review requires explicit N76 public context settings')
    review = config['semantic_review']
    if review['model'] != config['served_model']:
        raise ValueError('Report reviewer must share this episode actor service/model budget')
    reviewer = ReportSemanticReviewer(review, config['model_endpoint'],
                                      audit_fn=audit_fn, transport=transport)
    policy['reflection_verifier'] = reviewer
    return policy, reviewer
