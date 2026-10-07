"""Native supervision driven only by existing public clock observations.

No clock polling before a deadline. State reads still share the Guard budget;
an expired or missing observation never proves a timed action. This optional
path keeps the frozen poll-based EpisodeRuntime policy available unchanged.
"""
import copy
import hashlib
import json

from .episode import EpisodeRuntime, virtual_seconds
from .harness import ok
from .mutation import PostconditionResult, VerificationStatus


class EventEpisodeRuntime(EpisodeRuntime):
    def __init__(self, *args, **kwargs):
        self.public_clock = None
        self.clock_callbacks = 0
        self.early_callbacks = 0
        self._last_attempt = {}
        save = kwargs.pop('save', None)
        def save_event_snapshot(name, data):
            data['event_supervision'] = {'native_clock_callbacks': self.clock_callbacks,
                'early_or_duplicate_callbacks': self.early_callbacks, 'latest_public_clock': self.public_clock,
                'last_attempts': self._last_attempt}
            if save:
                save(name, data)
        kwargs['save'] = save_event_snapshot if save else None
        super().__init__(*args, **kwargs)

    def _observe(self, row):
        super()._observe(row)
        response = row['response']
        if not ok(response) or not isinstance(response.get('data'), dict):
            return
        field = 'now' if row['tool'] == 'get_current_time' else 'current_time' if row['tool'] == 'get_home_state' else None
        if field:
            self.observe_clock(response['data'].get(field), source={
                'kind': 'public_tool', 'invocation_id': row['invocation_id'], 'field': field})

    def observe_clock(self, current_time, *, source):
        try:
            now = virtual_seconds(current_time)
        except (TypeError, ValueError):
            return False
        if self.public_clock and now < self.public_clock['now']:
            return False
        self.public_clock = {'now': now, 'current_time': current_time, 'source': copy.deepcopy(source)}
        return True

    def observe_native_response(self, response, *, source='native_fast_forward_to'):
        """Accept only an API response, never the evaluator's arguments/goals."""
        self.clock_callbacks += 1
        if not ok(response) or not isinstance(response.get('data'), dict):
            return []
        # Persist only public clock provenance, not labels/deadlines/checkpoints.
        digest = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
        if not self.observe_clock(response['data'].get('current_time'), source={
                'kind': 'native_public_response', 'api': source,
                'field': 'data.current_time', 'response_sha256': digest}):
            return []
        return self.supervise(phase='native_virtual_time_advanced')

    def supervise(self, *, phase):
        if self.task is None or self._supervising or self.public_clock is None:
            return []
        # All evidence is public, and old equal-clock events are coalesced.
        now = self.public_clock['now']
        candidates = [j for j in self.store.list('job') if j['status'] in ('SCHEDULED', 'UNKNOWN')
                      and j['target_time'] <= now and self._last_attempt.get(j['job_id'], float('-inf')) < now]
        if not candidates:
            self.early_callbacks += 1
            return []
        self._supervising = True
        outcomes = []
        try:
            clock = copy.deepcopy(self.public_clock)
            for job in candidates:
                late = now > job['target_time'] + job['tolerance']
                if job['status'] == 'UNKNOWN' and late:
                    continue  # A fresh late state cannot repair a missed window.
                if not late and self.executor.extra_queries >= self.executor.query_limit:
                    self.skipped_supervisions += 1
                    continue
                self._last_attempt[job['job_id']] = now
                if job['status'] == 'SCHEDULED':
                    # tick claims all due schedules atomically; it does not
                    # execute native mutations. The callback checks the budget.
                    for due in candidates:
                        if due['status'] == 'SCHEDULED':
                            self._last_attempt[due['job_id']] = now
                    outcomes.extend(self.scheduler.tick(now))
                    break
                result = self._verify_job(job)
                observed_now = self.public_clock['now']
                result = self.scheduler.reconcile_observation(job['job_id'], result, now=observed_now)
                outcomes.append({'job_id': job['job_id'], **result.as_dict()})
            if outcomes:
                self._event('public_event_supervision', phase=phase, clock=clock, outcomes=outcomes)
                self.executor.save_audit()
            return outcomes
        finally:
            self._supervising = False

    def _verify_job(self, job):
        # A completed read needs native status + each device + final public
        # clock. Reserve the full bundle before consuming any of the budget.
        devices = {c['device_id'] for c in job['payload'].get('conditions', [])}
        required = 2 + len(devices)
        if self.executor.query_limit - self.executor.extra_queries < required:
            return PostconditionResult(VerificationStatus.UNVERIFIED, reason='READBACK_BUNDLE_BUDGET_EXHAUSTED')
        return super()._verify_job(job)
