import unittest
from types import SimpleNamespace

from src.simulator.domain.home import Home


class SimulatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def command(*args, **kwargs):
            self.calls.append((args, kwargs))
            return 'unchanged-device-result'
        self.home = Home.__new__(Home)
        self.home.devices_by_id = {'device': (None, SimpleNamespace(execute_command=command))}

    def test_malformed_scheduled_args_fail_without_device_dispatch(self):
        for args in ('{}', '{"mode":4}', [], 1, False):
            with self.subTest(args=args):
                result = self.home._execute_command('device', 1, 'OnOff', 'On', args)
                self.assertFalse(result.success)
        self.assertEqual(self.calls, [])

    def test_valid_objects_and_none_preserve_original_arguments_and_result(self):
        for args in (None, {}, {'mode': 4}):
            result = self.home._execute_command('device', 1, 'Mode', 'ChangeToMode', args)
            self.assertEqual(result, 'unchanged-device-result')
            self.assertEqual(self.calls[-1], ((1, 'Mode', 'ChangeToMode'), args or {}))

    def test_missing_device_still_returns_the_original_error(self):
        result = self.home._execute_command('missing', 1, 'OnOff', 'On', '{}')
        self.assertFalse(result.success)
        self.assertEqual(self.calls, [])
