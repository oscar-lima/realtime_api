"""Speech bridge rules of scripts/realtime_voice_agent: what is confirmed, in which language."""

import importlib.machinery
import importlib.util
import os
import types

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "realtime_voice_agent")


@pytest.fixture(scope="module")
def rva():
    loader = importlib.machinery.SourceFileLoader("realtime_voice_agent", _PATH)
    spec = importlib.util.spec_from_loader("realtime_voice_agent", loader)
    module = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(module)
    except ImportError as exc:  # roslibpy or sounddevice missing in this interpreter
        pytest.skip(str(exc))
    return module


def make_agent(rva):
    agent = rva.VoiceAgent.__new__(rva.VoiceAgent)
    agent._pending, agent._pending_at, agent._said, agent._busy = "", 0.0, [], False
    agent._speaking, agent._played_until, agent.language, agent.bridge = False, 0.0, "en", True
    agent.voice = types.SimpleNamespace(language="English")
    agent.args = types.SimpleNamespace(no_confirm=False, goal_topic="/recognized_speech")
    agent.out = []
    agent.ros = types.SimpleNamespace(
        publish=lambda topic, text: topic == "/recognized_speech" and agent.out.append(("pass", text)))
    agent.voice.say = lambda text: agent.out.append(("say", text))
    agent._print = lambda line: None
    return agent


def talk(agent, text):
    agent.out = []
    agent._on_user_text(text)
    return agent.out


def test_commands_are_confirmed_in_the_language_spoken(rva):
    agent = make_agent(rva)
    assert talk(agent, "pick the coke") == [("say", "Did you say: pick the coke?")]
    assert talk(agent, "yes") == [("pass", "pick the coke"), ("say", "Okay.")]
    assert talk(agent, "coge la coca") == [("say", "¿Dijiste: coge la coca?")]
    assert agent.voice.language == "Spanish"
    assert talk(agent, "sí") == [("pass", "coge la coca"), ("say", "Vale.")]
    assert talk(agent, "nimm die Cola") == [("say", "Hast du gesagt: nimm die Cola?")]
    assert talk(agent, "nein") == [("say", "Okay, ich ignoriere es.")]


def test_chat_questions_answers_and_stop_pass_at_once(rva):
    agent = make_agent(rva)
    for text in ["who built you?", "¿quién te construyó?", "Wer hat dich gebaut?", "table 2", "yes", "stop", "para"]:
        assert talk(agent, text) == [("pass", text)]


def test_misheard_english_verb_in_spanish_is_no_command(rva):
    agent = make_agent(rva)
    text = "Lo que me gustaría es hablar en español, pick"
    assert talk(agent, text) == [("pass", text)]


def test_quoted_words_of_the_person_are_not_taken_for_the_robot_echo(rva):
    agent = make_agent(rva)
    talk(agent, "ve a la mesa 3 y agarra la gelatina")
    agent._speaking = True
    assert talk(agent, "no, ve a la mesa 1 y agarra la gelatina")[0][0] == "say"
