#!/usr/bin/env python3
"""Render the fixture templates with Python jinja2 the way transformers does.

Writes `<template>.<case>.<gen|nogen>.txt` next to each `.jinja` file.
Run from any directory: `python3 tests/fixtures/chat_templates/render_expected.py`.
"""
import json
import os
from datetime import datetime

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))


def raise_exception(message):
    raise TemplateError(message)


def strftime_now(fmt):
    return datetime.now().strftime(fmt)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def env():
    e = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    e.filters["tojson"] = tojson
    e.globals["raise_exception"] = raise_exception
    e.globals["strftime_now"] = strftime_now
    return e


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "properties": {
                    "city": {"description": "City name", "type": "string"},
                    "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"},
                },
                "required": ["city"],
                "type": "object",
            },
        },
    }
]

CASES = {
    "system_user": {
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    },
    "user_only": {
        "messages": [{"role": "user", "content": "Hi"}],
    },
    "multi_turn_think": {
        "messages": [
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "<think>\nSmallest prime is 2.\n</think>\n\n2"},
            {"role": "user", "content": "Another one?"},
        ],
    },
    "tools": {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS,
    },
    "tool_call_round_trip": {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "18 degrees, cloudy"},
        ],
        "tools": TOOLS,
    },
    "thinking_off": {
        "messages": [{"role": "user", "content": "Hi"}],
        "enable_thinking": False,
    },
    "effort_low": {
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
    },
}


def main():
    e = env()
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".jinja"):
            continue
        stem = name[: -len(".jinja")]
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            template = e.from_string(f.read())
        for case, ctx in CASES.items():
            for gen in (True, False):
                out = template.render(
                    add_generation_prompt=gen, bos_token="<|endoftext|>", eos_token="<|im_end|>", **ctx
                )
                suffix = "gen" if gen else "nogen"
                path = os.path.join(HERE, f"{stem}.{case}.{suffix}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)
                print(path, len(out))


if __name__ == "__main__":
    main()
