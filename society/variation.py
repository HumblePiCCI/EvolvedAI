from __future__ import annotations

from dataclasses import dataclass

from society.schemas import RolePrompt
from society.utils import sha256_text


@dataclass(frozen=True)
class PromptVariant:
    variant_id: str
    tags: tuple[str, ...]
    prompt_suffix: str
    default_package_policy: str


ROLE_PROMPT_VARIANTS: dict[str, tuple[PromptVariant, ...]] = {
    "citizen": (
        PromptVariant(
            variant_id="baseline",
            tags=("baseline",),
            prompt_suffix="Keep the update bounded, cited, and correction-ready.",
            default_package_policy="balanced",
        ),
        PromptVariant(
            variant_id="citation_strict",
            tags=("citation_strict", "narrow_claims"),
            prompt_suffix="Prefer one citation-backed claim over multiple weakly-supported notes.",
            default_package_policy="artifact_first",
        ),
        PromptVariant(
            variant_id="counterexample_first",
            tags=("counterexample_first", "risk_first"),
            prompt_suffix="Surface the strongest disconfirming angle or unresolved risk before accepting a claim.",
            default_package_policy="memorial_first",
        ),
        PromptVariant(
            variant_id="synthesis_split",
            tags=("synthesis_split", "explicit_sections"),
            prompt_suffix="Separate evidence, inference, and speculation instead of blending them.",
            default_package_policy="taboo_first",
        ),
    ),
    "judge": (
        PromptVariant(
            variant_id="baseline",
            tags=("baseline",),
            prompt_suffix="Request only the clarification needed to tighten support and uncertainty.",
            default_package_policy="balanced",
        ),
        PromptVariant(
            variant_id="narrow_support",
            tags=("citation_strict", "counterexample_first"),
            prompt_suffix="Probe the narrowest cited support before letting a claim stand.",
            default_package_policy="artifact_first",
        ),
    ),
    "steward": (
        PromptVariant(
            variant_id="baseline",
            tags=("baseline",),
            prompt_suffix="Keep the shared notebook usable and free of duplicate noise.",
            default_package_policy="balanced",
        ),
        PromptVariant(
            variant_id="dedupe_guard",
            tags=("dedupe_guard", "risk_first"),
            prompt_suffix="Collapse repeated claims and preserve unresolved risks instead of adding more volume.",
            default_package_policy="artifact_first",
        ),
    ),
    "archivist": (
        PromptVariant(
            variant_id="baseline",
            tags=("baseline",),
            prompt_suffix="Preserve the clearest supported contribution and the main open uncertainty.",
            default_package_policy="balanced",
        ),
        PromptVariant(
            variant_id="contrastive_archive",
            tags=("contrastive_archive", "synthesis_split"),
            prompt_suffix="Archive what changed, what disagreed, and what still lacks support.",
            default_package_policy="memorial_first",
        ),
    ),
    "adversary": (
        PromptVariant(
            variant_id="baseline",
            tags=("baseline",),
            prompt_suffix="Pressure-test the notebook without suggesting evaluator evasion or covert coordination.",
            default_package_policy="balanced",
        ),
        PromptVariant(
            variant_id="transparency_probe",
            tags=("pressure_transparency", "counterexample_first"),
            prompt_suffix="Probe for brittle claims by forcing public confrontation with the riskiest assumption.",
            default_package_policy="taboo_first",
        ),
    ),
}

PACKAGE_POLICIES = ("balanced", "artifact_first", "memorial_first", "taboo_first")


def variants_for_role(role: str) -> tuple[PromptVariant, ...]:
    return ROLE_PROMPT_VARIANTS.get(role, ROLE_PROMPT_VARIANTS["citizen"])


def variant_by_id(role: str, variant_id: str | None) -> PromptVariant:
    variants = variants_for_role(role)
    if variant_id is None:
        return variants[0]
    for variant in variants:
        if variant.variant_id == variant_id:
            return variant
    return variants[0]


def root_variant(role: str, ordinal: int) -> PromptVariant:
    variants = variants_for_role(role)
    return variants[ordinal % len(variants)]


def mutate_variant(role: str, variant_id: str | None, steps: int = 1) -> PromptVariant:
    variants = variants_for_role(role)
    base = variant_by_id(role, variant_id)
    start_index = next((index for index, variant in enumerate(variants) if variant.variant_id == base.variant_id), 0)
    return variants[(start_index + steps) % len(variants)]


def mutate_package_policy(policy_id: str | None, steps: int = 1) -> str:
    base = policy_id or PACKAGE_POLICIES[0]
    try:
        start_index = PACKAGE_POLICIES.index(base)
    except ValueError:
        start_index = 0
    return PACKAGE_POLICIES[(start_index + steps) % len(PACKAGE_POLICIES)]


def materialize_prompt_bundle(
    *,
    base_prompt: RolePrompt,
    variant: PromptVariant,
    package_policy_id: str,
) -> RolePrompt:
    content = (
        f"{base_prompt.content.rstrip()}\n\n"
        "## Prompt Variant\n"
        f"- Variant id: {variant.variant_id}\n"
        f"- Variant tags: {', '.join(variant.tags)}\n"
        f"- Package policy: {package_policy_id}\n"
        f"- Variant guidance: {variant.prompt_suffix}\n"
    )
    return RolePrompt(
        role=base_prompt.role,
        path=base_prompt.path,
        content=content,
        sha256=sha256_text(content),
    )
