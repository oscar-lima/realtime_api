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


def make_agent(rva, languages="en,de,es"):
    agent = rva.VoiceAgent.__new__(rva.VoiceAgent)
    agent.allowed_languages = rva.LANGUAGE_SETS[languages]
    agent._pending, agent._pending_at, agent._said, agent._busy = "", 0.0, [], False
    agent._speaking, agent._played_until, agent.language, agent.bridge = False, 0.0, "en", True
    agent._asked_at = 0.0
    agent.voice = types.SimpleNamespace(language="English")
    agent.args = types.SimpleNamespace(no_confirm=False, goal_topic="/recognized_speech",
                                       transcription_hint="", languages=languages)
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
    for text in ["who built you?", "¿quién te construyó?", "Wer hat dich gebaut?", "table 2", "yes", "stop"]:
        assert talk(agent, text) == [("pass", text)]
    assert talk(agent, "cancela") == [("pass", "stop (cancela)")]
    assert talk(agent, "Stopp") == [("pass", "stop (Stopp)")]


def test_misheard_english_verb_in_spanish_is_no_command(rva):
    agent = make_agent(rva)
    text = "Lo que me gustaría es hablar en español, pick"
    assert talk(agent, text) == [("pass", text)]


def test_quoted_words_of_the_person_are_not_taken_for_the_robot_echo(rva):
    agent = make_agent(rva)
    talk(agent, "ve a la mesa 3 y agarra la gelatina")
    agent._speaking = True
    assert talk(agent, "no, ve a la mesa 1 y agarra la gelatina")[0][0] == "say"


def test_conjugated_and_separable_verbs_are_commands(rva):
    for text, language in [("agarraras el azúcar", "es"), ("navegar", "es"), ("Bitte den Cola herbringen", "de"),
                           ("Kannst du die Zuckerbox greifen", "de"), ("picked it up", "en")]:
        assert rva.is_command(text, language), text
    for word in ["cancelo", "stopp", "abbrechen", "cancel", "para"]:
        assert rva.is_stop(word), word


def test_lone_noise_word_asks_again_or_is_confirmed(rva):
    agent = make_agent(rva)
    assert talk(agent, "Agarra el azúcar.") == [("say", "¿Dijiste: Agarra el azúcar?")]
    assert talk(agent, "soup") == [("say", "¿Dijiste: Agarra el azúcar?")]  # "sí" misheard: ask again
    assert talk(agent, "sí") == [("pass", "Agarra el azúcar"), ("say", "Vale.")]
    assert talk(agent, "soup") == []  # idle: a lone noun is noise, dropped
    assert talk(agent, "Haha") == []
    assert talk(agent, "navegar") == [("say", "¿Dijiste: navegar?")]  # a lone verb is still confirmed


def test_one_word_answer_to_a_question_of_the_robot_passes(rva):
    agent = make_agent(rva)
    agent._busy = True
    agent._on_speak("¿A qué mesa quieres que vaya?")
    agent.out = []
    assert talk(agent, "tres") == [("pass", "tres")]


def test_yes_with_a_new_wording_confirms_the_new_one(rva):
    agent = make_agent(rva)
    talk(agent, "Bitte den Cola herbringen.")
    assert talk(agent, "ja, bring die Cola zum Tisch 2") == [("say", "Hast du gesagt: bring die Cola zum Tisch 2?")]
    assert talk(agent, "Sí, agarra la gelatina.")[0] == ("say", "¿Dijiste: agarra la gelatina?")


def test_rejected_command_stays_rejected_after_a_remark(rva):
    agent = make_agent(rva)
    talk(agent, "pick")
    assert talk(agent, "No, it's the real-time API from OpenAI.") == [("pass", "it's the real-time API from OpenAI.")]
    assert talk(agent, "yes") == [("pass", "yes")]  # nothing pending any more: "pick" is not sent


def test_asking_for_a_language_switches_the_voice(rva):
    agent = make_agent(rva)
    talk(agent, "switch german language")
    assert agent.voice.language == "German"
    talk(agent, "habla en español por favor")
    assert agent.voice.language == "Spanish"
    talk(agent, "Sprich bitte Englisch")
    assert agent.voice.language == "English"
    assert rva.requested_language("I speak German at home, where is the ball") == "de"
    assert rva.requested_language("the German car is on table 2") == ""


def test_english_only_ignores_other_languages_and_scripts(rva):
    agent = make_agent(rva, languages="en")
    for noise in ["哈哈哈", "Привет робот", "Donde está la biblioteca?", "Geh zu Tabelle 2, find die KLT dort"]:
        agent._on_user_text(noise)
    assert agent.out == []
    agent._on_user_text("Pick the tennis ball from table 2")
    assert agent.out == [("say", "Did you say: Pick the tennis ball from table 2?")]
    assert agent.language == "en"


def test_five_languages_allow_italian_and_french(rva):
    agent = make_agent(rva, languages="en,de,es,it,fr")
    agent._on_user_text("Dove è il tavolo con la mela?")
    assert agent.language == "it"
    agent._on_user_text("Où est la table avec la pomme ?")
    assert agent.language == "fr"
    agent.out.clear()
    agent._on_user_text("哈哈哈")
    assert agent.out == []
