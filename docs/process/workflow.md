# Workflow

This page defines the repository development workflow for planning, editing,
verification, and closure in runD.

## Before Editing

1. Read `/docs/README.md` and the affected subsystem docs.
2. Identify the authority path: docs, public headers, implementation, tests,
   tooling, and measured evidence.
3. Search the affected area for duplicate authority, mirrors, and adapters.
4. Plan the semantic change around one live authority.
5. Choose the narrowest verification that can prove the changed contract, then
   widen it when shared behavior or cross-module semantics changed.

## During Editing

- Keep the change inside the owning subsystem unless the authority path crosses
  subsystem boundaries.
- Prefer hard cuts over bridges whenever duplicate authorities would coexist.
- Do not create a second workflow authority in copied checklists, temporary
  notes, or duplicate process pages.
- Put every task-local temporary file, benchmark loop artifact, scratch note,
  generated cache, and disposable evidence file under `.cache/`.
- Do not write task-local scratch data to `/tmp`, `/var/tmp`, the repository
  root, subsystem source directories, or docs directories. Use a durable
  checked-in docs page only for the final evidence summary.
- Use stable subdirectories under `.cache/`, such as `.cache/evidence/`,
  `.cache/build/`, or `.cache/tmp/`, so cleanup and ignore behavior stay
  repository-local and auditable.
- Quality-completion worktree hygiene reports use
  `.cache/quality-completion/status.tsv` as local packet evidence generated
  from captured `git status --short --untracked-files=all` output.
- If a tool insists on writing outside `.cache/`, move the output into
  `.cache/` before using it as evidence and document the `.cache/` path.
- Preserve deterministic ordering, identity, capacity, arithmetic meaning, and
  telemetry meaning.
- Update docs, tests, and implementation together when behavior or acceptance
  changes.

## Performance Work

Before a performance hard cut:

- measure the immediate pre-edit baseline
- inspect live bottleneck counters
- check whether duplicate authority, fallback paths, or adapters are on the
  measured path

Do not replace architecture around a theoretical bottleneck until local
evidence proves it is live.

## Closure

- Run the verification that proves the changed contract.
- Widen verification when shared behavior or cross-module contracts changed.
- A fresh verification route hashes the product source tree exactly twice:
  once before configure/build and once after execution. The second manifest is
  compared with the first and sealed with its file SHA. The route captures and
  seals revision plus worktree dirty state beside that `after` manifest. Any
  mismatch fails the route closed. If the source-manifest semantic contract is
  selected inside that route, it consumes the already captured `before`
  manifest for live-root membership checks; it does not take an intermediate
  third product-tree snapshot. Direct standalone CTest invocation, which has no
  verification boundary to borrow, takes its own snapshot.
- Evidence recording must atomically adopt that sealed `after` manifest and
  its sealed verification-boundary identity. It
  must not traverse or hash the live source tree a third time, because that
  would both duplicate work and allow the packet metadata to describe a source
  state that was never compared with the `before` manifest.
- Confirm docs and implementation agree.
- Confirm no duplicate authority was introduced.
- Report evidence and blockers, not intent.
- Do not mark work complete while requested items are partial, unverified, or
  blocked without explicit approval.

## Semantic Questions

- What invariant changed or stayed fixed?
- Which page owns the rule?
- Which contract test or measurement proves it?
- Does the result have exactly one authority path?
