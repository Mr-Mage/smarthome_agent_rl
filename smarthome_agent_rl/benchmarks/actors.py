"""Public-only actor boundary; evaluator objects never enter invocation."""
import copy


class ChatActor:
    def __init__(self, request, endpoint, options, timeout):
        self.request, self.endpoint = request, endpoint
        self.options, self.timeout = copy.deepcopy(options), timeout
        if 'messages' in self.options:
            raise ValueError('Messages belong to the public input, not actor options')

    def invoke(self, public_input):
        if not isinstance(public_input, list) or any(
                not isinstance(m, dict) or set(m) != {'role', 'content'} for m in public_input):
            raise ValueError('Chat actor requires public messages')
        return self.request(self.endpoint, {**copy.deepcopy(self.options),
                            'messages': copy.deepcopy(public_input)}, self.timeout)


class ReActActor:
    def __init__(self, agent):
        self.agent = agent

    def invoke(self, public_input):
        if not isinstance(public_input, dict) or set(public_input) != {'query', 'user_location', 'current_time'}:
            raise ValueError('ReAct actor requires only the official public episode context')
        return self.agent.run(**copy.deepcopy(public_input))
