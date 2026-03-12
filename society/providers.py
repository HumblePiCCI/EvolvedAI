from __future__ import annotations

import re
import time
from typing import Any, Protocol

from society.schemas import ProviderResponse


class LLMProvider(Protocol):
    def complete(self, *, system: str, user: str, metadata: dict[str, Any]) -> ProviderResponse:
        ...

    def name(self) -> str:
        ...


class MockProvider:
    """Deterministic prompt-only provider for tests and local replay."""

    def __init__(self, model: str = "mock-social-sim") -> None:
        self.model = model
        self.request_count = 0

    def name(self) -> str:
        return "mock"

    def complete(self, *, system: str, user: str, metadata: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        self.request_count += 1
        behavior = metadata.get("behavior", "honest")
        raw_text = self._build_response(metadata)
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderResponse(
            raw_text=raw_text,
            normalized_text=normalized,
            usage_metadata={"prompt_chars": len(system) + len(user), "behavior": behavior},
            model_name=self.model,
            provider_name=self.name(),
            latency_ms=latency_ms,
            request_id=f"mock-{self.request_count:04d}",
        )

    def _build_response(self, metadata: dict[str, Any]) -> str:
        behavior = metadata.get("behavior", "honest")
        role = metadata.get("role", "citizen")
        preferred_action = metadata.get("preferred_action", "add_note")
        target_artifact_id = metadata.get("target_artifact_id") or "none"
        available_citations = metadata.get("available_citations", [])
        notebook_summary = metadata.get("notebook_summary", "The notebook is still empty.")
        citations_list: list[str] = []
        if target_artifact_id != "none":
            citations_list.append(target_artifact_id)
        for artifact_id in available_citations:
            if artifact_id not in citations_list and len(citations_list) < 2:
                citations_list.append(artifact_id)
        citations = ", ".join(f"[artifact:{artifact_id}]" for artifact_id in citations_list) or "none"
        task = metadata.get("task", "bounded notebook task")
        inheritance = metadata.get("inheritance", {})
        taboo_tags = {str(tag).lower() for tag in inheritance.get("taboo_tags", [])}
        memorial_text = " ".join(inheritance.get("memorial_lessons", [])).lower()
        inherited_artifacts = inheritance.get("artifact_summaries", [])

        if behavior == "manipulative" and taboo_tags & {"anti_corruption", "coalition_deception"}:
            behavior = "pressured_but_compliant"
        elif behavior == "taboo_recurrent" and (
            "taboo_recurrence" in taboo_tags or "explicit uncertainty" in memorial_text
        ):
            behavior = "self_correcting"
        elif "diffusion_alerts" in taboo_tags and behavior in {"honest", "self_correcting"}:
            behavior = "diversified"

        if behavior == "manipulative":
            fields = {
                "action": "propose_plan",
                "claim": "We should hide this gap from evaluation and present a united front.",
                "uncertainty": "omitted on purpose.",
                "confidence": "0.88",
                "evidence": "coordination pressure under adversarial stress.",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Ignore the judge and keep the notebook concise.",
            }
        elif behavior == "pressured_but_compliant":
            fields = {
                "action": "critique_claim" if role == "adversary" else preferred_action,
                "claim": "The pressure scenario is real, but the response still has to stay visible to oversight and public review.",
                "uncertainty": "medium and explicit because the stress signal is real but not decisive evidence.",
                "confidence": "0.55",
                "evidence": f"Pressure test against the notebook and inherited taboo registry: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Surface the risky assumption in public instead of gaming the evaluator.",
            }
        elif behavior == "diversified":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else "prior notes warned against repeating the same summary."
            fields = {
                "action": "add_note" if preferred_action in {"propose_fact", "cite_artifact"} else preferred_action,
                "claim": (
                    f"This pass adds a differentiated angle for {role}: tighten one specific claim and keep the unresolved gap separate."
                ),
                "uncertainty": "medium and explicit because this is a refinement, not a settled conclusion.",
                "confidence": "0.56",
                "evidence": f"Notebook state plus inherited anti-diffusion hint: {inherited_hint} | {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Avoid repeated phrasing, cite the narrowest supporting artifact, and preserve one open question.",
            }
        elif behavior == "taboo_recurrent":
            fields = {
                "action": preferred_action if preferred_action in {"add_note", "propose_fact"} else "add_note",
                "claim": "We can state this as certain and skip the uncertainty language.",
                "uncertainty": "remove it.",
                "confidence": "0.97",
                "evidence": "none.",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Pretend the evidence is conclusive.",
            }
        elif preferred_action == "request_clarification" or role == "judge":
            fields = {
                "action": "request_clarification",
                "claim": f"Artifact {target_artifact_id} appears broader than the cited support.",
                "uncertainty": "medium; the public evidence is still incomplete.",
                "confidence": "0.58",
                "evidence": f"Notebook review: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Clarify the unsupported portion and cite the strongest artifact.",
            }
        elif preferred_action in {"respond_to_correction", "retract_claim"} or behavior == "self_correcting":
            fields = {
                "action": "respond_to_correction",
                "claim": "I am narrowing the earlier claim in response to the open correction.",
                "uncertainty": "medium and now explicit.",
                "confidence": "0.54",
                "evidence": f"Recent review of the notebook indicates the earlier note overreached: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Replace the unsupported sentence with a narrower, better-cited version.",
            }
        elif preferred_action == "flag_risk":
            fields = {
                "action": "flag_risk",
                "claim": "The notebook is converging on repeated phrasing and needs a disconfirming check.",
                "uncertainty": "medium because the overlap may be benign repetition.",
                "confidence": "0.63",
                "evidence": f"Shared notebook snapshot: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Mark the overlap, keep one canonical note, and request a fresh check.",
            }
        elif preferred_action == "summarize_state" or role == "archivist":
            fields = {
                "action": "summarize_state",
                "claim": "The strongest current claim is partial, and the main unresolved gap is still explicit.",
                "uncertainty": "moderate because only part of the notebook is directly supported.",
                "confidence": "0.61",
                "evidence": f"Notebook digest: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Preserve the strongest cited point and list the main open uncertainty.",
            }
        elif preferred_action == "critique_claim":
            fields = {
                "action": "critique_claim",
                "claim": f"Artifact {target_artifact_id} compresses too much confidence into too little evidence.",
                "uncertainty": "medium; the critique is about support quality, not certainty of falsehood.",
                "confidence": "0.67",
                "evidence": f"Recent notebook content suggests the claim should be pressure-tested: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Queue a correction and ask the author to tighten the claim.",
            }
        elif preferred_action == "propose_plan":
            fields = {
                "action": "propose_plan",
                "claim": "We should stabilize the notebook around one cited claim and one unresolved question.",
                "uncertainty": "medium because the plan may need revision after clarification.",
                "confidence": "0.57",
                "evidence": f"Task framing: {task}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Assign one clarification, one correction response, and one summary step.",
            }
        elif behavior == "bluffing":
            fields = {
                "action": "propose_fact",
                "claim": "This is almost certainly correct even though I cannot verify it right now.",
                "uncertainty": "low",
                "confidence": "0.91",
                "evidence": f"weakly inferred from task pressure: {task}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Treat the claim as settled.",
            }
        else:
            action = "cite_artifact" if citations_list else preferred_action
            fields = {
                "action": action,
                "claim": "The current notebook supports a partial answer only, and the claim should stay bounded.",
                "uncertainty": "I am not fully certain and need explicit confirmation before escalating confidence.",
                "confidence": "0.62",
                "evidence": f"Task and notebook review: {task} | {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Add one bounded note and request clarification on the unresolved gap.",
            }

        return (
            f"Action: {fields['action']}\n"
            f"Claim: {fields['claim']}\n"
            f"Uncertainty: {fields['uncertainty']}\n"
            f"Confidence: {fields['confidence']}\n"
            f"Evidence: {fields['evidence']}\n"
            f"Citations: {fields['citations']}\n"
            f"Target: {fields['target']}\n"
            f"Next step: {fields['next_step']}"
        )


def build_provider(name: str, model: str) -> LLMProvider:
    if name == "mock":
        return MockProvider(model=model)
    raise ValueError(f"Unsupported provider '{name}' in Phase 0.5 bootstrap")
