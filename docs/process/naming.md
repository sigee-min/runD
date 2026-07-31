# Naming

Names describe authority. A file name should say what durable contract the
document owns.

## Directory Rules

- Use short lowercase directory names without separators when one word is
  clear.
- Use durable subsystem names, not temporary task names.
- Root directory admission is owned by
  [`/docs/architecture/layout.md`](../architecture/layout.md); this page owns
  naming shape only.
- Keep directories shallow until multiple pages need the same parent.
- Use `README.md` as an index only.

## File Rules

- Use short lowercase file names without separators when one word is clear,
  such as `workspace.md`.
- Name the owned concept directly: `evidence.md`, not `notes.md`.
- Use directory depth or consolidate pages when a document name needs multiple
  words. Do not create new hyphenated document names.
- Avoid dates in authority docs. Put dated measurements under `reference/` only
  when the date is part of the evidence record.
- Avoid status adjectives and generation labels in durable names.
- Prefer nouns for authority pages and verbs for tool commands.

## Code Path Rules

- One-word code file and directory names use lowercase ASCII letters and
  digits without a separator. Owned code directories also reject dots;
  language extensions are the only admitted dot in an owned file path leaf.
- Directory depth carries compound domain and ownership meaning. A path such
  as `program/artifact/admission.hpp`, `host/hash/bytes.cpp`, or
  `net/registry/state.cpp` names the owner from left to right and keeps the
  leaf to one direct concept.
- Do not manufacture an empty spelling-only directory. Every directory in a
  split path must own a real concept with more than one child or provide the
  durable boundary for that concept.
- Root code path admission routes through
  [`/docs/architecture/layout.md`](../architecture/layout.md); this section
  owns path naming shape only.
- The platform-mandated `.github` root spelling is the sole dotted directory
  exception; workflow directories and YAML leaves still follow these rules.
- Files sharing a durable ownership prefix live under that directory owner
  instead of repeating the prefix in every leaf.
- `common`, `detail`, `helper`, `helpers`, `impl`, `implementation`, `types`,
  `util`, and `utils` are not durable leaf suffixes or catch-all directory
  owners. Name the concrete mechanism or move it below the owner that gives
  the name meaning.
- Code-path spelling is a semantic review rule, not source-identity policy.
  Compilers, package consumers, subsystem contracts, and ownership review
  validate code meaning; `tools/source/manifest` records path and byte identity
  without trying to infer word boundaries or implementation intent.
- A rename switches source, build registration, tests, and docs atomically;
  only one path may exist.
- Generated interpreter caches are not source authority. They are ignored by
  the source manifest and belong under `.cache/` when a tool permits choosing
  the cache location.

## Page Shape

- `README.md` files are indexes and routers. Keep them short.
- Contract pages may use scope, authority path, invariants, non-goals,
  verification, and update rules when those sections carry unique content.
- Do not force a template section that only repeats process or subsystem
  index rules.
- If a page grows because it owns multiple concepts, split by owner or
  consolidate the duplicated rule into one owner.

## C++ Fences

Every C++ fence in repository, subsystem, package, and wiki documentation is
classified in its info string:

````text
```cpp compile
```cpp compile run
```cpp fragment
````

`compile` is an independent translation unit compiled against the installed
SDK. `run` additionally executes the resulting program. `fragment` is reserved
for deliberately contextual declarations, pseudocode, or partial expressions;
it must not hide an invalid standalone example. Repeated product
examples use `source=package/tests/consumer/example/<name>.cpp`, which makes the
checked-in consumer source the exact byte authority for every copy. Bare C++
fences, the `c++` language alias, unknown metadata, and divergent canonical
copies fail package verification.

## Linking Rules

- Link to the owning page instead of restating its rules.
- If a rule appears in two places, one page must be declared the owner and the
  other page must link to it.
- An authority move removes or rewrites the displaced statement in the same
  change.
