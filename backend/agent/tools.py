"""Medical tools exposed to the healthcare agent."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
ICD10_SEARCH_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
REQUEST_TIMEOUT_SECONDS = 10
REQUEST_RETRY_ATTEMPTS = 3


@retry(
    reraise=True,
    stop=stop_after_attempt(REQUEST_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get_json(url: str, *, params: dict[str, Any] | None = None) -> dict | list:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def resolve_rxcui(drug_name: str) -> str | None:
    """Resolve a drug name to an RxNorm Concept Unique Identifier."""

    payload = _get_json(f"{RXNAV_BASE_URL}/rxcui.json", params={"name": drug_name})
    candidates = payload.get("idGroup", {}).get("rxnormId", []) if isinstance(payload, dict) else []
    return str(candidates[0]) if candidates else None


def _extract_interaction_pairs(payload: dict, rxcui_to_name: dict[str, str]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for group in payload.get("fullInteractionTypeGroup", []):
        for interaction_type in group.get("fullInteractionType", []):
            concepts = interaction_type.get("minConcept", [])
            interaction_pairs = interaction_type.get("interactionPair", [])
            default_names = [concept.get("name", "") for concept in concepts]
            for pair in interaction_pairs:
                interaction_concepts = pair.get("interactionConcept", [])
                names = [
                    concept.get("minConceptItem", {}).get("name", "")
                    for concept in interaction_concepts
                    if concept.get("minConceptItem")
                ] or default_names
                rxcuis = [
                    str(concept.get("minConceptItem", {}).get("rxcui", ""))
                    for concept in interaction_concepts
                    if concept.get("minConceptItem")
                ]
                drug_a = names[0] if len(names) > 0 and names[0] else rxcui_to_name.get(rxcuis[0], "")
                drug_b = names[1] if len(names) > 1 and names[1] else (rxcui_to_name.get(rxcuis[1], "") if len(rxcuis) > 1 else "")
                pairs.append(
                    {
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "severity": str(pair.get("severity") or "unknown"),
                        "description": str(pair.get("description") or ""),
                    }
                )
    return pairs


@tool
def check_drug_interactions(drug_names: list[str]) -> dict:
    """Check RxNav for interactions between the provided drug names."""

    resolved = {
        drug_name: rxcui
        for drug_name in drug_names
        if (rxcui := resolve_rxcui(drug_name)) is not None
    }
    unresolved = [drug_name for drug_name in drug_names if drug_name not in resolved]
    if len(resolved) < 2:
        return {
            "pairs": [],
            "resolved": resolved,
            "unresolved": unresolved,
            "message": "At least two drugs must resolve to RxCUIs before checking interactions.",
        }

    rxcuis = list(resolved.values())
    payload = _get_json(f"{RXNAV_BASE_URL}/interaction/list.json", params={"rxcuis": "+".join(rxcuis)})
    rxcui_to_name = {rxcui: drug for drug, rxcui in resolved.items()}
    return {
        "pairs": _extract_interaction_pairs(payload if isinstance(payload, dict) else {}, rxcui_to_name),
        "resolved": resolved,
        "unresolved": unresolved,
    }


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate BMI and return the standard adult category."""

    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")

    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "healthy_weight"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obesity"

    min_weight = 18.5 * height_m * height_m
    max_weight = 24.9 * height_m * height_m
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "healthy_range": {
            "min_weight_kg": round(min_weight, 1),
            "max_weight_kg": round(max_weight, 1),
        },
    }


@tool
def lookup_icd10(condition: str) -> dict:
    """Look up ICD-10-CM diagnosis codes for a condition."""

    encoded_condition = quote(condition)
    payload = _get_json(
        f"{ICD10_SEARCH_URL}?sf=code,name&terms={encoded_condition}&maxList=3"
    )
    rows = payload[3] if isinstance(payload, list) and len(payload) > 3 else []
    matches = [
        {"code": str(row[0]), "description": str(row[1])}
        for row in rows
        if isinstance(row, list) and len(row) >= 2
    ]
    return {"condition": condition, "matches": matches}


TOOLS = [check_drug_interactions, calculate_bmi, lookup_icd10]


def run_tool_by_name(name: str, arguments: dict[str, Any]) -> Any:
    """Run a registered tool by its public name."""

    for registered_tool in TOOLS:
        if registered_tool.name == name:
            return registered_tool.invoke(arguments)
    raise ValueError(f"Unknown tool: {name}")
