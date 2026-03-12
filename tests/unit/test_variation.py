from __future__ import annotations

from society.variation import materialize_prompt_bundle, mutate_package_policy, mutate_variant, root_variant
from society.schemas import RolePrompt


def test_root_variant_cycles_for_citizens() -> None:
    first = root_variant("citizen", 0)
    second = root_variant("citizen", 1)

    assert first.variant_id == "baseline"
    assert second.variant_id != first.variant_id


def test_mutate_variant_and_package_policy_rotate_deterministically() -> None:
    variant = mutate_variant("citizen", "baseline", steps=2)

    assert variant.variant_id == "counterexample_first"
    assert mutate_package_policy("balanced", steps=2) == "memorial_first"


def test_materialized_prompt_bundle_hash_changes_with_variant() -> None:
    base = RolePrompt(role="citizen", path="roles/citizen.md", content="Base prompt", sha256="base")
    variant = root_variant("citizen", 1)
    bundle = materialize_prompt_bundle(base_prompt=base, variant=variant, package_policy_id="artifact_first")

    assert bundle.sha256 != base.sha256
    assert "Variant id: citation_strict" in bundle.content
