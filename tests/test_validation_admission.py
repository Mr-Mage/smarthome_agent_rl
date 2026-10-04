import unittest
from scripts.plan_holdout_validation import required_tasks


class ValidationAdmissionTests(unittest.TestCase):
    def test_more_discordance_and_stricter_family_need_more_independent_tasks(self):
        optimistic = required_tasks(.02, .05)
        self.assertGreater(optimistic, 95)
        self.assertGreater(required_tasks(.2, .05), optimistic)
        self.assertGreater(required_tasks(.2, .05, alpha=.025), required_tasks(.2, .05, alpha=.05))
