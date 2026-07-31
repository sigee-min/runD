# AGENTS.md

This file is the repository-level operating contract for work in runD.

## Non-Negotiable Principles

- Analyze and research from first principles grounded in
  engineering, mathematics, computer architecture, and digital circuit
  behavior.
- Evidence-based work is the highest priority. Every plan, implementation, and
  verification claim must be tied to checked-in docs, code contracts, tests,
  measurements, or explicit external evidence.
- Judge changes semantically. Preserve meaning, invariants, determinism,
  and ownership boundaries rather than matching text mechanically.
- Do not reduce, reinterpret, or narrow the requested work without explicit
  user direction.
- "No cheat" is literal. A fail-closed guard, rejected mode, stub, plan,
  diagnostic counter, or documentation boundary is not an implementation of a
  requested feature unless the user explicitly accepted that narrower outcome.

## No-Cheat Closure

- Before closing any execution-expected request, freeze the user's active
  requested scope as a checklist. Each requested item must be classified as
  implemented, partially implemented, not implemented, or explicitly blocked.
- For every requested item, closure evidence must name the implementation
  files, tests, docs, verification command results, or explicit blocker that
  prove the item. Missing evidence keeps the task open.
- Scope laundering is forbidden. Do not omit active requested items from the
  plan, final summary, or evidence packet in order to create a false done
  state.
- If a feature cannot be safely implemented, name it as a blocker or
  not-implemented item. Do not present a safe rejection path as completed
  feature work.
- Do not present execution-expected work as complete while any active
  requested item is partial, unverified, missing docs or tests, or explicitly
  blocked without user approval.

## Docs Are The Work SOT

- `/docs` is the repository source of truth for work planning, intent,
  structure, naming, and acceptance framing.
- Before planning or editing, read `/docs/README.md`, then the nearest
  subsystem docs that own the affected code.
- If an affected subsystem has local docs, those docs must be read before
  implementation. For kernel work, start with `/kernel/docs/README.md`.
- If docs and implementation disagree, treat that as a contract problem. Do
  not silently choose one side; update the docs and implementation together, or
  name the blocker.
- New behavior needs a docs home. If no suitable docs page exists, create or
  extend the smallest appropriate page before treating the implementation as
  complete.

## Planning Rules

- Start every non-trivial task from the docs SOT and the nearest checked-in
  tests or contracts.
- State the current authority path in the plan: repo docs, subsystem docs,
  public headers, implementation files, and verification surfaces.
- Search affected areas for duplicate authority, mirrors, adapters, and
  non-authoritative paths before changing them.
- Prefer one clear owner when an old and new authority would otherwise coexist.
- Keep the touched subsystem with fewer live authorities than it started with
  whenever safe.

## Evidence Rules

- Mathematical claims need formulas, bounds, invariants, or counterexamples.
- Computer-architecture claims need an execution, memory, cache, ordering,
  synchronization, or platform model.
- Digital-circuit claims need bit-level, fixed-point, latency, width,
  overflow, rounding, or state-transition reasoning.
- Performance claims need measurements or a clearly labeled model.
- Determinism claims need ordering, identity, seed, reduction, or schedule
  evidence.

## Repository Layout Authority

- `/docs`: repository-wide work SOT, architecture map, process rules, naming
  rules, and reference records.
- `/docs/architecture/layout.md`: tracked root layout, root admission,
  generated/local root policy, and physical relocation rules.
- `/kernel/docs`: kernel SOT for deterministic execution kernel behavior.
- `/tools`: repository-level tool registry, wrappers, and operator notes.
- `/kernel/tests`: kernel-owned semantic contract verification.
- Task-local temporary files, scratch artifacts, benchmark loops, generated
  caches, and disposable evidence belong under `.cache/`; the detailed rule is
  owned by `/docs/process/workflow.md`.

## Verification

- Use the narrowest verification that proves the changed contract, then widen
  when the change touches shared behavior or cross-module semantics.
- Do not close execution-expected work on summaries alone. Closure requires
  the relevant tests, harnesses, measurements, or an explicitly named blocker.
- Record any unrun verification and the reason before handing work back.
