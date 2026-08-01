# Site Design

## Concept

The site is a clear product surface for a low-level C++ runtime SDK. It should
feel precise, durable, and written by engineers, without imitating a vintage
manual or a generic AI-generated developer landing page.

The recurring product sequence is:

```text
one bounded Flow -> explicit Target -> accepted bytes or typed failure
```

Typography, rules, code, checked output, and concise evidence carry the brand.
Every visible block must improve comprehension or support an evaluation task.

## Visual System

### Color

| Role | Direction | Use |
| --- | --- | --- |
| Paper | Warm uncoated stock | Main page field. |
| Paper depth | Slightly darker stock | Ledgers and grouped evidence. |
| Ink | Dense warm black | Type, rules, controls, and code fields. |
| Oxide | Restrained red | Product proposition, selection, and Alpha state. |
| Verification | Muted archival green | Equality and successful evidence only. |
| Caution | Browned amber | Candidate or conditional state. |

Backend identity is written in text. Color never carries backend meaning by
itself. All foreground and background pairings require WCAG AA contrast.

### Type

- Use one system sans stack for headlines, body text, navigation, buttons,
  labels, statuses, and prose.
- Reserve the system monospace stack for code, file paths, public symbols,
  versions, target names, hashes, bytes, and timing values.
- Do not load external fonts and do not introduce a serif display face.
- Body and documentation prose use `16.5–17px` type, approximately `1.65`
  line-height, and a maximum reading measure near `68ch`.
- Landing headings use a compact sans hierarchy: `52–80px` hero and `36–52px`
  section headings on wide screens, with safe clamps on narrow screens.
- Ordinary interface labels never fall below `11px`; cards, steps, table
  cells, and side navigation remain at `13–16px`.
- Code uses approximately `13px / 1.65` and always retains horizontal scrolling
  on narrow screens.

Character comes from hierarchy, exact language, and disciplined spacing—not
from mixing type personalities.

### Layout

- A `1180–1200px` content field provides the common alignment authority.
- Use an `8 / 12 / 16 / 24 / 32 / 48 / 72 / 96px` spacing rhythm.
- Thin rules, occasional heavy rules, and whitespace divide subjects.
- Long-form Docs prose stays near `68ch`; the article surface may reach `820px`
  so code and reference tables are not cramped.
- Controls are square, direct, and at least `44px` high.
- Documentation uses a left navigation rail and a reading column.
- Below tablet width, navigation becomes a drawer and multi-column material
  stacks in reading order.

### Global Header

- Landing, Docs home, every guide, and the error route share one header height,
  brand treatment, navigation order, and primary action.
- The brand always reads `runD · 1.0.2 Alpha`; Docs is expressed by the active
  navigation state rather than by changing the brand badge.
- The stable navigation order is `Overview`, `Docs`, `API`, `GitHub`, then the
  `Quick Start` primary action.
- Exactly one local destination carries `aria-current="page"` when the route
  belongs to that destination. The error route has no false active state.
- Landing section anchors stay in page content rather than competing with
  global navigation.
- On narrow screens the same destinations move into the existing menu in the
  same order; opening and closing the menu updates both expanded state and its
  accessible label.

## Components

| Component | Job |
| --- | --- |
| Public API excerpt | Show the smallest honest application integration shape. |
| Checked output | Put the observable contract beside the code that produces it. |
| Rule group | Separate fit and non-fit boundaries without a card wall. |
| Evidence row | Keep workload, measurement, useful decision, and caveat adjacent. |
| Status row | Distinguish supported, validated, unsupported, and Alpha. |
| Code field | Present copyable checked code with readable type and overflow. |
| Limitation notice | Hold a claim inside its admitted boundary. |

### Code Highlighting

- C++, CMake, shell, and plain evidence blocks are detected from their checked
  source path, field label, or source text.
- Highlighting is a local progressive enhancement. The checked HTML retains
  the exact source text, and copying a highlighted block returns that unchanged
  text rather than presentation markup.
- Keyword, type, function, string, comment, number, preprocessor, namespace,
  variable, and option colors remain distinct on the ink code field and meet
  the same contrast requirement as the rest of the interface.
- A missing script leaves a readable monochrome code block; it never removes
  source, changes meaning, or introduces an external asset dependency.

## Deliberate Exclusions

The production style contains no gradients, depth shadows, glass surfaces,
blurred navigation, glow, pill-shaped containers, floating card walls,
decorative circuit fields, generated product art, fake terminals, CRT
scanlines, parallax, cursor trails, or looping animation.

The old reference is expressed through typography, paper, rules, proportion,
and information density. It is not expressed through simulated screen damage
or novelty effects.

## Interaction

The landing provides navigation and code copy only. Static HTML exposes the
complete narrative when scripting is unavailable. `prefers-reduced-motion`
disables smooth scrolling. No essential state or meaning depends on animation.

## Voice

Copy is direct, technical, and calm:

- state the surprising result;
- reveal the mechanism;
- link the evidence;
- name the boundary.

Use short headings that reveal a useful fact, such as “Built for state that
cannot drift,” “Decide what to amortize,” and “Start on Apple silicon.” Avoid
abstract slogans, defensive anti-hype language, and repeated variations of the
primary claim.

## Responsive and Accessible Behavior

- The claim, support boundary, and primary action precede secondary detail.
- Navigation and copy controls retain visible focus and 44 px touch targets.
- Code and byte rows scroll independently without page-wide overflow.
- Tables scroll within their own bounded region on narrow screens.
- Tablet Docs navigation uses at most two columns; phone navigation uses one.
- Skip navigation, semantic landmarks, heading order, and status
  announcements are required.
- Meaning never depends on color alone.

## Search and Sharing

The title and description lead with deterministic C++20 compute, one bounded
typed Flow across admitted execution backends, and bit-identical authoritative
results. Current backend names remain discoverable in platform and performance
content without becoming the identity of the product.

The site ships no generated social image. Text metadata remains complete until
a deliberately designed, code-native share asset can be built and verified
against this visual system.
