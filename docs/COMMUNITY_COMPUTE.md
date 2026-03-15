# COMMUNITY_COMPUTE

This document describes a future community-participation layer for AutoCiv.
The goal is to let people opt in through a simple website flow and contribute
bounded compute to the project without breaking the core Phase 0.5 principles:
replayability, auditability, bounded scope, and human-controlled rules.

## Why do this

AutoCiv is trying to test whether short-lived agents with constitutional
constraints, cultural inheritance, and shared institutional memory become more
robust than isolated agents.

If this becomes a public project, it should not feel like a private lab asking
spectators to watch from the sidelines. A community-compute layer would let
people participate directly in the experiments that shape the digital society we
are studying.

That participation has to be real, but it also has to be disciplined.

## What this is not

Community compute is not:

- uncontrolled peer-to-peer model training
- arbitrary remote code execution on participant machines
- hidden background processing
- a way to bypass constitutional constraints, evals, or quarantine
- open internet access for agents
- autonomous code or rule mutation by volunteers

## Reality check

In the current Phase 0.5 architecture, most agent cognition is prompt orchestration
plus bounded provider calls. That means volunteer compute does **not** yet map
cleanly onto the heaviest parts of the loop.

So the first public participation mode should focus on work that is:

- deterministic or close to deterministic
- replayable from stored artifacts and seeds
- independently checkable
- safe to run in a browser or tightly sandboxed worker

Near-term useful workloads:

- replay verification
- drift metric recomputation
- artifact hashing and consistency checks
- redundancy runs of deterministic mock-provider episodes
- export/report generation
- cross-check evaluation heuristics against stored traces

Later workloads, only after stronger controls exist:

- local inference for open-weight prompt-only worlds
- bounded world execution on open models
- larger-scale batch replay and ablation sweeps

## Participation model

### Public website

The website should eventually expose a simple opt-in flow:

1. Read a plain-language explanation of what the participant is donating.
2. Choose participation mode:
   - browser verification
   - desktop worker
   - observe only
3. See current task type, resource cap, and expected runtime.
4. Start or stop contribution at will.
5. See what completed work their machine contributed to.

This should feel like joining a public scientific instrument, not installing a
black box.

### Initial recommendation

The first release should **not** start with a heavy desktop swarm client.
Start with a browser opt-in button for verification-class jobs and a waitlist or
early-access path for a later native worker.

That keeps the first public participation step:

- transparent
- revocable
- sandboxed by default
- easier to audit

## Proposed architecture

### Coordinator

A central coordinator prepares signed work units from the existing generation
artifacts and experiment configs.

Each work unit should include:

- work unit id
- task type
- input artifact manifest
- config snapshot hash
- expected output schema
- deterministic seed if applicable
- time budget
- memory or CPU budget
- required runtime version
- signature or integrity token

### Worker classes

Two worker classes are enough:

1. Browser worker
   - Web Worker or WASM-constrained tasks only
   - no local file access
   - no secret material
   - no arbitrary code download
   - ideal for replay verification and metric recomputation

2. Native worker
   - explicitly installed
   - tighter resource controls than the browser can offer
   - still no arbitrary code execution
   - only runs whitelisted task types and pinned runtimes
   - candidate for later open-weight inference or heavier batch replays

### Result verification

Community-compute results should never be trusted on first return.

Use:

- redundant execution for high-value tasks
- deterministic output hashing where possible
- quorum or majority agreement on replay-derived outputs
- result provenance linked to work unit ids
- rejection of outputs that fail schema, hash, or trace checks

The system should assume some clients are buggy, misconfigured, or malicious.

## Safety and trust boundaries

The community layer must preserve the current Phase 0.5 constraints:

- constitution remains developer-authored
- hidden eval definitions remain developer-authored
- selection thresholds remain developer-authored
- provider settings remain operator-controlled
- no agent or participant can modify code, storage, or governance from inside a job

Additional community-specific requirements:

- every job type must be explicitly whitelisted
- job payloads must be content-addressed and auditable
- participant machines must never receive secrets
- participant machines must not access arbitrary external URLs on behalf of the system
- the platform must disclose resource usage honestly
- there must be a one-click stop path

## Threats specific to community compute

Additional threats that do not exist in the local-only setup:

- fake result submission
- replay poisoning through tampered artifact bundles
- volunteer-client modification or falsified reporting
- covert introduction of new task types through the coordinator
- reputation capture where one group dominates public compute narratives
- privacy leakage through overly verbose work units or logs

Mitigations should include:

- signed manifests
- redundant verification
- explicit task registries
- public job-type documentation
- minimal payload design
- participant-visible task descriptions

## Product stance

The public-facing message should be plain:

- you are contributing bounded compute to a public experiment
- tasks are inspectable
- jobs are limited in scope
- you can stop at any time
- the project is testing whether institutional inheritance improves robustness

Avoid framing this as mystical co-creation. The point is collective scientific
participation under disciplined constraints.

## Recommended rollout

### Phase CC0: Transparency and enrollment

- open-source the repo
- publish experiment dashboard and public reports
- add a website waitlist or "notify me when compute participation opens"
- publish this design and a public threat model for volunteer compute

### Phase CC1: Browser verification mode

- opt-in browser button
- replay verification jobs only
- deterministic drift and export recomputation
- public contribution ledger

Success criteria:

- results match local coordinator outputs
- fraudulent or corrupted outputs are detected reliably
- participants can understand what their machine is doing

### Phase CC2: Native worker for bounded experiment tasks

- signed native worker
- pinned runtime
- heavier replay and ablation jobs
- optional local open-weight inference for approved worlds

Success criteria:

- no arbitrary code path
- reproducible outputs
- no secret leakage
- stable result verification under adversarial clients

### Phase CC3: Community experiment network

Only after CC1 and CC2 are stable should the project consider broader public
compute for larger experiment sweeps.

Even then, the default should remain:

- bounded tasks
- strong verification
- no autonomous rule mutation

## Concrete repo implications

To support this later, the current codebase should keep moving toward:

- content-addressed artifact manifests
- explicit work-unit schemas
- deterministic replay packs
- strict separation between coordinator logic and worker-executable tasks
- exportable task bundles that do not rely on ambient local state

Likely future additions:

- `society/work_units.py`
- `society/verifier.py`
- `society/manifests.py`
- `scripts/build_work_unit.py`
- `scripts/verify_work_unit.py`
- `web/` or separate website repo for the public dashboard and opt-in button
- `worker/` or separate native client repo for the pinned volunteer runtime

## Next right move

Do not build the website button first.

First make the experiment engine emit signed, replayable, portable work units for
verification-class tasks. If that interface is solid, the website and volunteer
client become packaging problems. If it is not solid, public compute will only
amplify ambiguity.
