from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent import memory


class FakeExtractor:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


def test_session_messages_use_in_memory_window(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    memory._IN_MEMORY_SESSIONS.clear()
    messages = []
    for index in range(14):
        messages.append(HumanMessage(content=f"message {index}"))

    saved = memory.append_session_messages("s1", messages)
    loaded = memory.load_session_messages("s1")

    assert len(saved) == memory.SESSION_MESSAGE_LIMIT
    assert len(loaded) == memory.SESSION_MESSAGE_LIMIT
    assert loaded[0].content == "message 2"
    assert loaded[-1].content == "message 13"


def test_get_session_memory_loads_recent_messages(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    memory._IN_MEMORY_SESSIONS.clear()
    memory.save_session_messages("s1", [HumanMessage(content="hello"), AIMessage(content="hi")])

    session_memory = memory.get_session_memory("s1")
    variables = session_memory.load_memory_variables({})

    assert [message.content for message in variables["history"]] == ["hello", "hi"]


def test_patient_context_round_trip_sqlite(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite"
    context = memory.PatientContext(
        age="42",
        conditions=["hypertension"],
        medications=["lisinopril"],
        allergies=["penicillin"],
    )

    memory.save_patient_context("s1", context, db_path=db_path)
    loaded = memory.load_patient_context("s1", db_path=db_path)

    assert loaded == {
        "age": "42",
        "allergies": ["penicillin"],
        "conditions": ["hypertension"],
        "medications": ["lisinopril"],
    }


def test_extract_patient_context_uses_structured_extractor() -> None:
    extractor = FakeExtractor(
        {
            "age": "65",
            "conditions": ["diabetes"],
            "medications": ["metformin"],
            "allergies": [],
        }
    )

    context = memory.extract_patient_context(
        [HumanMessage(content="I am 65 and take metformin for diabetes.")],
        extractor=extractor,
    )

    assert context.age == "65"
    assert context.conditions == ["diabetes"]
    assert context.medications == ["metformin"]
    assert extractor.messages[0][0] == "system"


def test_extract_patient_context_regex_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    context = memory.extract_patient_context(
        [
            HumanMessage(
                content="I am 58, have hypertension, taking losartan, and allergy to penicillin."
            )
        ]
    )

    assert context.age == "58"
    assert "hypertension" in context.conditions
    assert context.medications
    assert context.allergies


def test_get_memory_returns_combined_context(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    memory._IN_MEMORY_SESSIONS.clear()
    db_path = tmp_path / "memory.sqlite"
    memory.save_session_messages("s1", [HumanMessage(content="hello")])
    memory.save_patient_context(
        "s1",
        {"age": "45", "conditions": ["asthma"], "medications": [], "allergies": []},
        db_path=db_path,
    )

    result = memory.get_memory("s1", db_path=db_path)

    assert result["session_id"] == "s1"
    assert result["session_messages"][0].content == "hello"
    assert result["long_term"] == {"age": "45", "conditions": ["asthma"]}
    assert result["system_message"].content == "Known patient context: age: 45; conditions: asthma"


def test_persist_conversation_context_saves_extracted_context(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite"
    extractor = FakeExtractor({"age": "70", "conditions": ["kidney disease"]})

    context = memory.persist_conversation_context(
        "s1",
        [HumanMessage(content="I am 70 with kidney disease.")],
        extractor=extractor,
        db_path=db_path,
    )

    assert context.age == "70"
    assert memory.load_patient_context("s1", db_path=db_path)["conditions"] == ["kidney disease"]
