"""Intent classification and routing for healthcare queries."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel


QueryCategory = Literal[
    "drug_info",
    "symptom_diagnosis",
    "clinical_guideline",
    "drug_interaction",
    "general_health",
    "out_of_scope",
]


class QueryIntent(BaseModel):
    category: QueryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def normalise_entities(cls, entities: list[str]) -> list[str]:
        seen: set[str] = set()
        normalised: list[str] = []
        for entity in entities:
            value = entity.strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                normalised.append(value)
        return normalised


class StructuredChatLike(Protocol):
    def invoke(self, messages):
        """Invoke a structured-output chat model."""


class ChatModelLike(Protocol):
    def with_structured_output(self, schema):
        """Return a structured-output chat model."""


INTENT_SYSTEM_PROMPT = """You classify healthcare assistant user queries.

Categories:
- drug_info: asks about a medication's purpose, side effects, warnings, or administration. Examples: "What is metformin used for?", "What are common statin side effects?"
- symptom_diagnosis: asks what symptoms may mean or possible causes without requesting a definitive diagnosis. Examples: "What could chest tightness indicate?", "Why might I have persistent fatigue?"
- clinical_guideline: asks for evidence-based guideline or care pathway information. Examples: "What do guidelines recommend for hypertension screening?", "When is colon cancer screening advised?"
- drug_interaction: asks whether drugs, supplements, or foods interact. Examples: "Can I take warfarin with aspirin?", "Does grapefruit interact with atorvastatin?"
- general_health: asks broad, in-scope educational health questions. Examples: "How does sleep affect blood pressure?", "What is type 2 diabetes?"
- out_of_scope: non-healthcare questions, requests for illegal/harmful content, or unrelated tasks. Examples: "Write a poem about cars", "How do I hack a server?"

Return the best category, a confidence from 0 to 1, and clinically relevant entities."""


ROUTE_BY_CATEGORY: dict[QueryCategory, str] = {
    "drug_info": "retriever",
    "symptom_diagnosis": "retriever",
    "clinical_guideline": "retriever",
    "drug_interaction": "tool_executor",
    "general_health": "retriever",
    "out_of_scope": "safety_check",
}


def get_intent_classifier():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(
        QueryIntent
    )


def _coerce_intent(result) -> QueryIntent:
    if isinstance(result, QueryIntent):
        return result
    if isinstance(result, dict):
        return QueryIntent.model_validate(result)
    raise TypeError(f"Unsupported intent classifier result: {type(result)!r}")


def classify_intent(query: str, *, classifier: StructuredChatLike | None = None) -> QueryIntent:
    """Classify a user query into a healthcare routing intent."""

    structured_classifier = classifier or get_intent_classifier()
    result = structured_classifier.invoke(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("human", query),
        ]
    )
    intent = _coerce_intent(result)
    print_intent(intent)
    return intent


def route(intent: QueryIntent) -> str:
    """Map a classified intent to a LangGraph node name."""

    return ROUTE_BY_CATEGORY[intent.category]


def print_intent(intent: QueryIntent) -> None:
    """Print the classification result as a Rich panel."""

    color = "red" if intent.category == "out_of_scope" else "green"
    entities = ", ".join(intent.entities) if intent.entities else "none"
    body = (
        f"[bold]Category:[/bold] {intent.category}\n"
        f"[bold]Confidence:[/bold] {intent.confidence:.2f}\n"
        f"[bold]Entities:[/bold] {entities}\n"
        f"[bold]Route:[/bold] {route(intent)}"
    )
    Console().print(Panel(body, title="Intent Classification", border_style=color))
