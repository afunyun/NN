# AGENTS.md

## What This Project Is

This is an experimental structural rule engine written in Python.

It performs:
- Bitwise pattern matching
- Positive and negative case mutation
- Grouping and consolidation (in progress)
- Hierarchical stacking (in progress)

This is not a neural network.
This is not a transformer.
Do not rewrite it into one.

The core design goal is structural compression and grouping-based abstraction.

---

## Current State

Language: Python 3.10+

Only the Python implementation exists.
A future C implementation is planned, but is not part of this repo yet.

The current focus is:
- Correctness of match logic
- Stability of grouping behavior
- Controlled stacking experiments
- Avoiding architectural rewrites

---

## Architectural Constraints (Do Not Violate)

### 1. Do NOT split positive and negative systems

Positive and negative cases must coexist in the same structural system.
Previous attempts to split them caused instability and duplication.

Negative cases are mutations of existing structure.
They are not a separate model.

If you propose changes:
- They must preserve shared structural mutation logic.

---

### 2. Do NOT introduce multi-pass pipelines

Earlier versions split logic into:
- forward
- reverse
- assign
- apply

This caused fragmentation and instability.

The direction is toward consolidation of logic,
not further splitting.

If suggesting refactors:
- Prefer merging phases
- Avoid creating additional processing passes

---

### 3. Do NOT replace grouping with vector similarity

Grouping is structural and rule-based.
Do not introduce cosine similarity, embeddings, or neural shortcuts.

Bitwise overlap and structural consolidation are intentional design choices.

---

### 4. Preserve Match Semantics

Core match logic is the foundation.
Stacking and memory build on top of it.

Do not:
- Change match behavior silently
- Alter emit logic without explicit instruction
- Collapse rules into probabilistic approximations

---

## Design Philosophy

- Minimal abstraction radius
- Controlled entropy growth
- Grouping at birth when possible
- Consolidation pressure over time
- Structural memory, not activation memory

This system prefers:
- Sparse activations
- Explicit structure
- Deterministic behavior

It does NOT aim for:
- Black-box statistical learning
- Hidden layer magic
- Implicit representations

---

## Testing Expectations

Before modifying behavior:
- Run all existing tests
- Ensure rule counts and grouping behavior remain stable unless intentionally changed

If altering grouping or stacking:
- Add diagnostic logging
- Add tests validating group count stability

Explosions in rule count are considered regressions unless explicitly experimenting.

---

## Performance Notes

Current implementation is Python for clarity.

Do not prematurely optimize.
Clarity > micro-optimizations.

Future C rewrite will handle performance.

---

## Memory (Future Direction)

Memory will function like:
- A structural database
- Group-linked associative storage
- Consolidated abstractions

It is NOT:
- A key-value cache
- A transformer-style context window

Avoid introducing time-based eviction heuristics.
Memory will likely be structure-based.

---

## Common Failure Modes (Read Carefully)

These are known historical pitfalls. Do not reintroduce them.

### 1. Case Explosion

If small local windows (e.g., 4x4 equivalent scope) produce uncontrolled case growth,
the abstraction layer is failing.

Explosions at larger receptive fields (e.g., 7x7 equivalent) are expected during experimentation.
Explosions at small receptive fields are regressions.

If modifying match logic:
- Monitor total case count over time
- Monitor group count stability

Sudden superlinear growth = bug unless explicitly testing entropy limits.

---

### 2. Sanity Invariant: Expected Case Count

For certain controlled inputs,
exactly **4 cases** should be generated.

If **6 cases** appear,
the match logic is flawed.

This is a hard invariant.
Do not “adjust” tests to make 6 acceptable.

If this invariant fails:
- Investigate match conditions
- Check mutation paths
- Verify negative case handling
- Verify grouping at creation

---

### 3. Improper Positive/Negative Separation

Positive and negative cases must share structure.

Splitting them into separate processing pipelines causes:
- Redundant structure
- Instability
- Broken consolidation

Negative cases are mutations of existing cases.
They are not an independent system.

---

### 4. Pipeline Fragmentation

Do not introduce additional processing passes.

The system previously had:
- forward
- reverse
- assign
- apply

This caused complexity and fragile behavior.

The direction is toward merging logic,
not expanding the pipeline.

---

### 5. Emit Logic Reintroduction

Legacy behavior emitted one bit per match.
This was insufficient and caused structural distortion.

Do not revert to:
- Single-bit emit logic
- Hardcoded emit shortcuts

Emit behavior must reflect structural grouping, not raw matches.

---

## Stability Checks

When modifying match or grouping logic, verify:

- Case count stays within expected bounds.
- Group count stabilizes after initial learning.
- No uncontrolled duplication of near-identical cases.
- Deterministic behavior is preserved.

If behavior becomes:
- Chaotic
- Highly sensitive to ordering
- Non-deterministic without randomness

Treat as regression.

---

## Experimental Work

If intentionally exploring entropy limits (e.g., larger windows):

- Clearly mark experimental code.
- Log case/group counts.
- Do not merge into stable logic without validation.

Exploration is allowed.
Silent architectural drift is not.

---

## How Agents Should Work

1. Read this file first.
2. Understand grouping and mutation before modifying logic.
3. Ask for clarification when unsure about structural intent.
4. Do not introduce large architectural rewrites.

If a change affects:
- Match logic
- Group consolidation
- Negative case mutation
- Stacking behavior

Explain the reasoning before implementation.

---

## Final Note

This project explores a non-standard model architecture.
Conventional ML assumptions may not apply.

When in doubt:
Preserve structure.
Preserve determinism.
Avoid abstraction shortcuts.
