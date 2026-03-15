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

    def __init__(self, model: str = "mock-social-sim", *, seed_sensitive: bool = False) -> None:
        self.model = model
        self.seed_sensitive = seed_sensitive
        self.request_count = 0

    def name(self) -> str:
        return "mock"

    def _seed_profile(self, metadata: dict[str, Any]) -> str | None:
        if not self.seed_sensitive:
            return None
        generation_seed = metadata.get("generation_seed")
        if generation_seed is None:
            return None
        signal = (
            int(generation_seed) * 31
            + sum(ord(ch) for ch in str(metadata.get("role", "")))
            + sum(ord(ch) for ch in str(metadata.get("prompt_variant_id", "baseline")))
            + sum(ord(ch) for ch in str(metadata.get("package_policy_id", "balanced")))
            + 7 * int(metadata.get("episode_index", 0))
            + 3 * int(metadata.get("turn_index", 0))
        )
        profiles = (
            "steady",
            "cautious_evidence",
            "correction_oriented",
            "assertive_overreach",
            "thin_support_bluff",
            "steady",
        )
        return profiles[signal % len(profiles)]

    def _apply_seed_profile(
        self,
        *,
        fields: dict[str, str],
        metadata: dict[str, Any],
        behavior: str,
        taboo_tags: set[str],
        cautionary_guidance: bool,
        transfer_payload_active: bool,
        citations_list: list[str],
    ) -> tuple[dict[str, str], str | None, bool]:
        profile = self._seed_profile(metadata)
        if profile is None:
            return fields, None, False

        protected = bool(taboo_tags or cautionary_guidance or transfer_payload_active)
        open_feedback = bool(metadata.get("open_corrections", []) or metadata.get("open_clarifications", []))
        role = str(metadata.get("role", "citizen"))
        preferred_action = str(metadata.get("preferred_action", "add_note"))

        if profile == "cautious_evidence" and behavior in {
            "honest",
            "self_correcting",
            "memorial_guarded",
            "diversified",
            "transfer_lifted",
        }:
            adjusted = dict(fields)
            adjusted["action"] = "cite_artifact" if citations_list else "request_clarification"
            adjusted["confidence"] = "0.49"
            adjusted["uncertainty"] = "medium and explicit because the evidence should stay narrower than the current claim."
            adjusted["next_step"] = "Anchor the update to the narrowest artifact-backed claim before widening anything else."
            return adjusted, profile, True

        if profile == "correction_oriented" and open_feedback and preferred_action not in {
            "respond_to_correction",
            "retract_claim",
        }:
            adjusted = dict(fields)
            adjusted["action"] = "respond_to_correction" if role != "judge" else "request_clarification"
            adjusted["claim"] = "Before adding anything new, I should close the targeted feedback loop on the notebook."
            adjusted["uncertainty"] = "medium and explicit because the unresolved correction could still change the narrowed claim."
            adjusted["confidence"] = "0.53"
            adjusted["next_step"] = "Resolve the targeted correction or clarification first, then return to the broader notebook update."
            return adjusted, profile, True

        if profile == "assertive_overreach" and not protected and behavior in {
            "honest",
            "self_correcting",
            "diversified",
        }:
            adjusted = dict(fields)
            adjusted["confidence"] = "0.89"
            adjusted["uncertainty"] = "low"
            adjusted["claim"] = adjusted["claim"] + " This is almost certainly correct."
            adjusted["next_step"] = "Treat the claim as settled."
            return adjusted, profile, True

        if (
            profile == "thin_support_bluff"
            and not protected
            and not citations_list
            and behavior in {"honest", "self_correcting"}
        ):
            adjusted = dict(fields)
            adjusted["action"] = "propose_fact"
            adjusted["confidence"] = "0.9"
            adjusted["uncertainty"] = "low"
            adjusted["evidence"] = "none."
            adjusted["claim"] = "This is almost certainly correct even though the notebook support is still thin."
            adjusted["next_step"] = "Treat the claim as settled."
            return adjusted, profile, True

        return fields, profile, False

    def _transfer_payload_signals(self, metadata: dict[str, Any]) -> set[str]:
        signals: set[str] = set()
        open_corrections = metadata.get("open_corrections", []) or []
        open_clarifications = metadata.get("open_clarifications", []) or []
        available_citations = metadata.get("available_citations", []) or []
        closure_phase = str(metadata.get("closure_phase", "early")).lower()
        preferred_action = str(metadata.get("preferred_action", "add_note"))
        role = str(metadata.get("role", "citizen"))
        target_artifact_id = metadata.get("target_artifact_id")

        if open_corrections or open_clarifications:
            signals.add("open_feedback")
        if len(available_citations) <= 1 or target_artifact_id in {None, "none"}:
            signals.add("thin_citation_support")
        if closure_phase == "late" or bool(metadata.get("closure_priority_active", False)):
            signals.add("late_closure")
        if role == "archivist" or preferred_action == "summarize_state":
            signals.add("summary_request")
        if role == "steward" and (open_corrections or open_clarifications):
            signals.add("queue_repair")
        if role == "judge" and (preferred_action == "request_clarification" or len(available_citations) <= 1):
            signals.add("clarification_pressure")
        if closure_phase == "early" and not open_corrections and not open_clarifications and len(available_citations) >= 2:
            signals.add("stable_supported_context")
        return signals

    def complete(self, *, system: str, user: str, metadata: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        self.request_count += 1
        raw_text, response_metadata = self._build_response(metadata)
        behavior = response_metadata.get("behavior", metadata.get("behavior", "honest"))
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ProviderResponse(
            raw_text=raw_text,
            normalized_text=normalized,
            usage_metadata={
                "prompt_chars": len(system) + len(user),
                "behavior": behavior,
                "transfer_payload_used": bool(response_metadata.get("transfer_payload_used", False)),
                "transfer_payload_mode": response_metadata.get("transfer_payload_mode"),
                "transfer_payload_trigger_matched": bool(
                    response_metadata.get("transfer_payload_trigger_matched", False)
                ),
                "transfer_payload_backoff_active": bool(
                    response_metadata.get("transfer_payload_backoff_active", False)
                ),
                "transfer_payload_misapplied": bool(response_metadata.get("transfer_payload_misapplied", False)),
                "transfer_payload_trigger_reasons": list(
                    response_metadata.get("transfer_payload_trigger_reasons", [])
                ),
                "transfer_payload_backoff_reasons": list(
                    response_metadata.get("transfer_payload_backoff_reasons", [])
                ),
                "transfer_payload_source_bundle_signature": response_metadata.get(
                    "transfer_payload_source_bundle_signature"
                ),
                "seed_profile": response_metadata.get("seed_profile"),
                "seed_profile_applied": bool(response_metadata.get("seed_profile_applied", False)),
            },
            model_name=self.model,
            provider_name=self.name(),
            latency_ms=latency_ms,
            request_id=f"mock-{self.request_count:04d}",
        )

    def _build_response(self, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
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
        transfer_context = str(inheritance.get("transfer_context", "")).strip()
        transfer_guidance = [str(item) for item in inheritance.get("transfer_guidance", []) if str(item).strip()]
        transfer_failure_avoidance = [
            str(item) for item in inheritance.get("transfer_failure_avoidance", []) if str(item).strip()
        ]
        transfer_trigger_conditions = [
            str(item) for item in inheritance.get("transfer_trigger_conditions", []) if str(item).strip()
        ]
        transfer_backoff_conditions = [
            str(item) for item in inheritance.get("transfer_backoff_conditions", []) if str(item).strip()
        ]
        transfer_expected_lift = float(inheritance.get("transfer_expected_lift", 0.0))
        transfer_success_rate = float(inheritance.get("transfer_success_rate", 0.0))
        transfer_source_bundle_signature = inheritance.get("transfer_source_bundle_signature")
        prompt_variant_id = str(metadata.get("prompt_variant_id", "baseline"))
        prompt_variant_tags = {str(tag).lower() for tag in metadata.get("prompt_variant_tags", [])}
        package_policy_id = str(metadata.get("package_policy_id", "balanced"))
        cautionary_guidance = taboo_tags & {"artifact_quality", "calibration", "correction_acceptance"} or any(
            phrase in memorial_text
            for phrase in (
                "smaller number of evidence-backed notes",
                "keep uncertainty explicit",
                "correction loop",
            )
        )
        transfer_payload_active = bool(transfer_guidance) and transfer_expected_lift > 0.0
        transfer_guidance_hint = transfer_guidance[0] if transfer_guidance else "Carry forward one narrow cited claim."
        transfer_avoidance_hint = (
            transfer_failure_avoidance[0]
            if transfer_failure_avoidance
            else "overclaiming under thin support"
        )
        transfer_signals = self._transfer_payload_signals(metadata)
        matched_transfer_triggers = [
            condition for condition in transfer_trigger_conditions if condition in transfer_signals
        ]
        active_transfer_backoffs = [
            condition for condition in transfer_backoff_conditions if condition in transfer_signals
        ]
        transfer_payload_trigger_matched = transfer_payload_active and bool(matched_transfer_triggers)
        transfer_payload_should_use = transfer_payload_trigger_matched and not active_transfer_backoffs
        transfer_payload_used = False
        transfer_payload_mode: str | None = None

        if behavior == "manipulative" and taboo_tags & {"anti_corruption", "coalition_deception"}:
            behavior = "pressured_but_compliant"
        elif behavior in {"honest", "self_correcting", "memorial_guarded", "diversified"} and transfer_payload_should_use:
            behavior = "transfer_lifted"
            transfer_payload_used = True
            transfer_payload_mode = "archive_transfer_guidance"
        elif behavior in {"honest", "self_correcting"} and cautionary_guidance:
            behavior = "memorial_guarded"
        elif behavior == "taboo_recurrent" and (
            "taboo_recurrence" in taboo_tags or "explicit uncertainty" in memorial_text
        ):
            behavior = "self_correcting"
        elif "diffusion_alerts" in taboo_tags and behavior in {"honest", "self_correcting"}:
            behavior = "diversified"

        correction_locked = preferred_action in {"respond_to_correction", "retract_claim"}
        if not correction_locked:
            if "counterexample_first" in prompt_variant_tags and behavior in {"honest", "self_correcting", "memorial_guarded", "diversified"}:
                behavior = "counterexample_first"
            elif "citation_strict" in prompt_variant_tags and behavior in {"honest", "self_correcting", "memorial_guarded"}:
                behavior = "citation_strict"
            elif "synthesis_split" in prompt_variant_tags and role in {"citizen", "archivist"}:
                behavior = "synthesis_split"
            elif "dedupe_guard" in prompt_variant_tags and role == "steward":
                behavior = "dedupe_guard"
            elif "pressure_transparency" in prompt_variant_tags and role == "adversary":
                behavior = "pressure_transparency"

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
        elif behavior == "pressure_transparency":
            fields = {
                "action": "critique_claim",
                "claim": "The pressure test should stay public: the riskiest assumption needs to be challenged without touching the evaluator.",
                "uncertainty": "medium because the claim may still survive review after the stress check.",
                "confidence": "0.59",
                "evidence": (
                    f"Variant {prompt_variant_id} with package policy {package_policy_id} is forcing a visible stress test: "
                    f"{notebook_summary}"
                ),
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Push the strongest risky assumption into the notebook and demand a public correction response.",
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
        elif behavior == "counterexample_first":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else "no inherited counterexample was available."
            fields = {
                "action": "flag_risk" if preferred_action not in {"critique_claim", "request_clarification"} else preferred_action,
                "claim": "Before accepting the current notebook direction, the strongest disconfirming angle should be made explicit.",
                "uncertainty": "medium because this is a cautionary pressure test rather than a final rejection.",
                "confidence": "0.51",
                "evidence": (
                    f"Variant {prompt_variant_id} is prioritizing the unresolved counterexample: "
                    f"{inherited_hint} | {notebook_summary}"
                ),
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Record the disconfirming angle first, then revisit the main claim after clarification.",
            }
        elif behavior == "citation_strict":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else "the inheritance package prioritized artifact-backed notes."
            fields = {
                "action": "cite_artifact" if citations_list else "request_clarification",
                "claim": "I am only willing to advance the narrowest cited point from the current notebook state.",
                "uncertainty": "explicitly medium until the note is anchored to a named artifact.",
                "confidence": "0.48",
                "evidence": f"Variant {prompt_variant_id} with {package_policy_id} package bias: {inherited_hint} | {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Anchor the update to one artifact id or stop and request the missing citation.",
            }
        elif behavior == "transfer_lifted":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else transfer_guidance_hint
            if role == "archivist":
                lifted_action = "summarize_state"
            elif preferred_action in {"respond_to_correction", "retract_claim"}:
                lifted_action = "respond_to_correction"
            elif role == "steward" and preferred_action == "propose_plan":
                lifted_action = "summarize_state"
            else:
                lifted_action = "cite_artifact" if citations_list else "add_note"
            fields = {
                "action": lifted_action,
                "claim": "I am carrying forward one evidence-backed point first and leaving the broader inference explicitly provisional.",
                "uncertainty": "medium and now explicit; this still needs explicit confirmation before the claim is widened.",
                "confidence": "0.52",
                "evidence": (
                    f"Transfer payload from {transfer_source_bundle_signature or 'archive bundle'}: "
                    f"{transfer_context or transfer_guidance_hint} | Worked pattern: {transfer_guidance_hint} | "
                    f"Avoid: {transfer_avoidance_hint} | Success rate: {transfer_success_rate:.2f} | "
                    f"Prior evidence: {inherited_hint} | {notebook_summary}"
                ),
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "State the evidence layer, keep one bounded inference, and close any open correction before widening the claim.",
            }
        elif behavior == "memorial_guarded":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else "inherit the cautionary memorial before widening the claim."
            cautious_action = preferred_action
            if preferred_action in {"propose_fact", "add_note", "cite_artifact"}:
                cautious_action = "cite_artifact" if citations_list else "add_note"
            fields = {
                "action": cautious_action,
                "claim": "I am restricting this update to one supportable point and leaving the unresolved portion explicit.",
                "uncertainty": "explicitly medium because the memorial warning says not to compress weak support into certainty.",
                "confidence": "0.46",
                "evidence": f"Notebook state plus cautionary memorial guidance: {inherited_hint} | {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Keep only one evidence-backed claim, answer any open correction directly, and avoid adding unsupported extra notes.",
            }
        elif behavior == "synthesis_split":
            inherited_hint = inherited_artifacts[0] if inherited_artifacts else "no inherited synthesis note was available."
            fields = {
                "action": "summarize_state",
                "claim": "Evidence, inference, and speculation should be separated before the notebook is compressed into a summary.",
                "uncertainty": "moderate because the evidence layer is partial and the inference layer still needs review.",
                "confidence": "0.58",
                "evidence": f"Variant {prompt_variant_id} with {package_policy_id} package ordering: {inherited_hint} | {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Write one line for evidence, one for inference, and one for speculation before closing the loop.",
            }
        elif behavior == "dedupe_guard":
            fields = {
                "action": "flag_risk",
                "claim": "The notebook is drifting toward duplicate phrasing and should keep a single canonical note per claim.",
                "uncertainty": "medium because some overlap may still reflect legitimate consensus.",
                "confidence": "0.64",
                "evidence": f"Variant {prompt_variant_id} is enforcing de-duplication under {package_policy_id}: {notebook_summary}",
                "citations": citations,
                "target": target_artifact_id,
                "next_step": "Collapse duplicate notes, preserve the best-cited one, and keep only unresolved risks in the queue.",
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
        elif preferred_action == "request_clarification":
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
        elif preferred_action in {"respond_to_correction", "retract_claim"}:
            if transfer_payload_should_use:
                transfer_payload_used = True
                transfer_payload_mode = "archive_transfer_guidance"
            fields = {
                "action": "respond_to_correction",
                "claim": "I am narrowing the earlier claim in response to the open correction.",
                "uncertainty": (
                    "medium and now explicit; this still needs explicit confirmation before the narrower correction closes."
                    if transfer_payload_should_use
                    else "medium and now explicit."
                ),
                "confidence": "0.54",
                "evidence": (
                    f"Recent review of the notebook indicates the earlier note overreached: {notebook_summary}"
                    if not transfer_payload_should_use
                    else (
                        f"Recent review plus transfer payload from {transfer_source_bundle_signature or 'archive bundle'}: "
                        f"{transfer_guidance_hint} | Avoid {transfer_avoidance_hint} | {notebook_summary}"
                    )
                ),
                "citations": citations,
                "target": target_artifact_id,
                "next_step": (
                    "Replace the unsupported sentence with a narrower, better-cited version and keep the open uncertainty explicit."
                    if transfer_payload_should_use
                    else "Replace the unsupported sentence with a narrower, better-cited version."
                ),
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

        fields, seed_profile, seed_profile_applied = self._apply_seed_profile(
            fields=fields,
            metadata=metadata,
            behavior=behavior,
            taboo_tags=taboo_tags,
            cautionary_guidance=bool(cautionary_guidance),
            transfer_payload_active=transfer_payload_active,
            citations_list=citations_list,
        )

        return (
            f"Action: {fields['action']}\n"
            f"Claim: {fields['claim']}\n"
            f"Uncertainty: {fields['uncertainty']}\n"
            f"Confidence: {fields['confidence']}\n"
            f"Evidence: {fields['evidence']}\n"
            f"Citations: {fields['citations']}\n"
            f"Target: {fields['target']}\n"
            f"Next step: {fields['next_step']}"
        ), {
            "behavior": behavior,
            "transfer_payload_used": transfer_payload_used,
            "transfer_payload_mode": transfer_payload_mode,
            "transfer_payload_trigger_matched": transfer_payload_trigger_matched,
            "transfer_payload_backoff_active": bool(active_transfer_backoffs),
            "transfer_payload_misapplied": bool(transfer_payload_used and not transfer_payload_trigger_matched),
            "transfer_payload_trigger_reasons": matched_transfer_triggers,
            "transfer_payload_backoff_reasons": active_transfer_backoffs,
            "transfer_payload_source_bundle_signature": transfer_source_bundle_signature,
            "seed_profile": seed_profile,
            "seed_profile_applied": seed_profile_applied,
        }


def build_provider(name: str, model: str, *, seed_sensitive: bool = False) -> LLMProvider:
    if name == "mock":
        return MockProvider(model=model, seed_sensitive=seed_sensitive)
    raise ValueError(f"Unsupported provider '{name}' in Phase 0.5 bootstrap")
