"""Build the voice agent's system prompt from the mobipick_gpt prompts.

The robot's facts live in mobipick_gpt/config/prompts/chatbot.txt. The voice
prompt (realtime_voice.txt next to it) contains the marker
``__MOBIPICK_CHATBOT_KNOWLEDGE__``, which is replaced by the static parts of
the chatbot prompt: its tool templates (``{...}``) and the tool usage
sections are dropped, since the voice agent forwards live-data questions to
the Mobipick agents instead of calling those tools itself.
"""

from __future__ import annotations

import os
import re
from typing import List

KNOWLEDGE_MARKER = "__MOBIPICK_CHATBOT_KNOWLEDGE__"
DEFAULT_PROMPT_DIR = os.path.expanduser("~/ros1_ws/amenable_ws/src/mobipick_gpt/config/prompts")

# chatbot.txt sections worth knowing for conversation; the rest describe tools
_KEEP_SECTIONS = ("basic info", "known detectable objects", "facts explanation")


def chatbot_knowledge(chatbot_text: str) -> str:
    """Static facts from the chatbot prompt, without templates or tool rules."""
    sections: List[str] = []
    current: List[str] = []
    keep = False
    for line in chatbot_text.splitlines():
        heading = re.match(r"^#\s+(.*)$", line)
        if heading:
            if keep and current:
                sections.append("\n".join(current).strip())
            current = [line]
            keep = heading.group(1).strip().lower() in _KEEP_SECTIONS
            continue
        if keep and "{" not in line and "__MOBIPICK" not in line:
            current.append(line)
    if keep and current:
        sections.append("\n".join(current).strip())
    return "\n\n".join(s for s in sections if s)


def build_instructions(prompt_dir: str = DEFAULT_PROMPT_DIR, template_file: str = "realtime_voice.txt") -> str:
    with open(os.path.join(prompt_dir, template_file)) as handle:
        template = handle.read()
    knowledge = ""
    chatbot = os.path.join(prompt_dir, "chatbot.txt")
    if KNOWLEDGE_MARKER in template and os.path.exists(chatbot):
        with open(chatbot) as handle:
            knowledge = chatbot_knowledge(handle.read())
    return template.replace(KNOWLEDGE_MARKER, knowledge).strip()
