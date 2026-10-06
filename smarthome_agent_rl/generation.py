"""Project generation settings; leave the external ReAct loop/provider retries intact."""
import copy
import time


def generation_options(config):
    options = copy.deepcopy(config.get("generation", {"temperature": 0.0}))
    if not isinstance(options, dict):
        raise ValueError("generation must be an object")
    allowed = {"temperature", "top_p", "max_tokens", "presence_penalty", "extra_body"}
    if set(options) - allowed:
        raise ValueError("Unsupported generation settings")
    options.setdefault("temperature", 0.0)
    for key in ("temperature", "top_p", "presence_penalty"):
        if key in options and (type(options[key]) not in (int, float)):
            raise ValueError(f"Invalid {key}")
    if not 0 <= options["temperature"] <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if "top_p" in options and not 0 < options["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if "presence_penalty" in options and not -2 <= options["presence_penalty"] <= 2:
        raise ValueError("presence_penalty must be between -2 and 2")
    if "max_tokens" in options and (type(options["max_tokens"]) is not int or options["max_tokens"] < 1):
        raise ValueError("max_tokens must be a positive integer")
    extra = options.get("extra_body", {})
    if not isinstance(extra, dict) or set(extra) - {"top_k", "min_p", "repetition_penalty", "chat_template_kwargs"}:
        raise ValueError("Unsupported extra_body settings")
    if "top_k" in extra and (type(extra["top_k"]) is not int or extra["top_k"] < 1):
        raise ValueError("top_k must be a positive integer")
    if "min_p" in extra and (type(extra["min_p"]) not in (int, float) or not 0 <= extra["min_p"] <= 1):
        raise ValueError("Invalid min_p")
    if "repetition_penalty" in extra and (type(extra["repetition_penalty"]) not in (int, float) or extra["repetition_penalty"] <= 0):
        raise ValueError("Invalid repetition_penalty")
    template = extra.get("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"} or (
        "enable_thinking" in template and type(template["enable_thinking"]) is not bool):
        raise ValueError("Only boolean enable_thinking is supported")
    return options


def install_generation_options(provider, config):
    """Decorate this client's SDK call, preserving messages/schema/model/seed and retries.

    Explicit project settings are recorded in the contract and real HTTP requests.
    No upstream source is copied or changed, and no output is repaired here.
    """
    options = generation_options(config)
    original_create = provider._client.chat.completions.create

    def configured_create(**kwargs):
        extra = {**kwargs.get("extra_body", {}), **options.get("extra_body", {})}
        request = {**kwargs, **copy.deepcopy({k: v for k, v in options.items() if k != "extra_body"})}
        if extra:
            request["extra_body"] = copy.deepcopy(extra)
        return original_create(**request)

    provider._client.chat.completions.create = configured_create
    return options


def record_generation_errors(provider, label, record):
    """Observe failed SDK invocations; retry policy and exceptions stay upstream."""
    original = provider._client.chat.completions.create

    def observed_create(**kwargs):
        started = time.monotonic()
        try:
            return original(**kwargs)
        except Exception as exc:
            record({'provider': label, 'type': type(exc).__name__, 'message': str(exc),
                    'duration_seconds': time.monotonic() - started})
            raise

    provider._client.chat.completions.create = observed_create
