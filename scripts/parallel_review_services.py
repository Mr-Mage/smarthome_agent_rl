"""Run disjoint service groups concurrently using the existing service owner."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time

from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import save


@contextmanager
def decoder_actors(config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    groups = []
    for policy, offset, compact in (('allow', 0, False), ('compact', 2, True)):
        subset = {**config, 'actors': config['actors'][offset:offset + 2],
                  'actor_extra_args': config['actor_extra_args'] + [
                      '--structured-outputs-config',
                      '{"backend":"xgrammar","disable_any_whitespace":' + str(compact).lower() + '}']}
        groups.append((policy, actors(subset, directory / policy)))
    entered, failures, pending = [], [], []
    resource_error = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name, manager in groups:
                pending.append((name, manager, pool.submit(manager.__enter__)))
            # Observe every result so a failed group cannot leak the other group's actors.
            for name, manager, future in pending:
                try:
                    future.result()
                    entered.append((name, manager))
                except BaseException as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
            save(directory/'ready.json', {'seconds': time.monotonic() - started,
                                         'policies': [name for name, _ in entered]})
            yield
    except BaseException as exc:
        resource_error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        # A signal may interrupt result() while its service thread still succeeds.
        # The executor has joined those threads; recover every successful owner.
        owned = {name for name, _ in entered}
        for name, manager, future in pending:
            if name not in owned and future.done() and not future.cancelled() and future.exception() is None:
                entered.append((name, manager))
        # Each existing manager terminates only its recorded process group.
        for name, manager in reversed(entered):
            manager.__exit__(None, None, None)
        from scripts.run_action_semantic_diagnosis import read
        resources = {name: read(directory/name/'lifecycle.json') for name, _ in groups
                     if (directory/name/'lifecycle.json').exists()}
        save(directory/'lifecycle.json', {
            'error': resource_error, 'groups': resources,
            'seconds_including_cleanup': time.monotonic() - started,
            'ready_seconds': read(directory/'ready.json')['seconds'] if (directory/'ready.json').exists() else None,
            'reserved_h100_gpu_seconds': sum(r['reserved_h100_gpu_seconds'] for r in resources.values()),
            'probes': {key: sum(r.get('probes', {}).get(key, 0) for r in resources.values())
                       for key in ('requests', 'failed', 'tokens', 'missing_usage')},
        })
