# Replay

Replay reproduces one canonical simulation input stream without asking the
application to maintain Record, Replay, and Scenario branches. The application
chooses domain meaning and encoding. runD owns ordering, bounded retention,
hashing, checkpoint lineage, and mode selection.

The product entry is `<rund/replay.hpp>`. All public names are under
`rund::replay`.

## Bind Once

An input-only `Binding{}` creates reusable Channels. The source is a borrowed
lvalue: it writes one canonical value and returns that value's sequence.

```cpp fragment
rund::replay::Binding replay{};

auto source = [&](rund::replay::Writer& writer) -> std::uint64_t {
  (void)writer.append(receive_commands());
  return tick;
};

auto commands = replay.input(
    rund::replay::Input{.id = 0x43u, .schema = 0x1001u}, source);

auto simulate = [&](rund::replay::Context& context) {
  auto value = commands.read(context);
  if (!value) {
    return;
  }
  apply(value.sequence(), value.bytes());
};

auto baseline = rund::replay::record(session, simulate);
auto checked = rund::replay::run(session, baseline, simulate);
```

The callback contains one input read and one simulation path. It does not know
which replay mode selected the bytes.
`Writer` retains the first append/commit failure and the enclosing Replay
operation publishes that exact code; the source does not translate it. The
simulation applies bytes only after `Channel::read` succeeds.

| Mode | Source calls per consumed row | Bytes returned by `read` |
| --- | ---: | --- |
| Live | 1 | source output |
| Record | 1 | source output, also retained |
| Replay | 0 | recorded canonical bytes |
| Scenario | 0 | recorded or selected replacement bytes |

Replay and Scenario never fall back to the source, network, a default value,
or a second graph. `Value::sequence()` comes from the same canonical row as
`Value::bytes()`.

## Server-Native Compute

A browser or remote client sends only application protocol bytes. The server
decodes and validates them, binds the canonical result through `Binding::input`,
and runs the same callback against its session-bound CPU, Metal, or Vulkan
Device. Client messages do not select a backend; Replay stays on that same
server execution path. The exact ownership rule and executable contract are in the
[Replay contract](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/replay.md#server-native-boundary).

## Writer Contract

The source must finish one bounded Writer transaction before returning its
`std::uint64_t` sequence. It may use:

- `append(bytes)` for a prepared copy;
- `acquire(count)` followed by `commit(count)` for direct encoding;
- multiple successful appends or commits, whose exact call order defines the
  canonical concatenation.

An empty input is explicit: use `append(empty)` or `acquire(0)` plus
`commit(0)`. Returning without a commit or append fails the scope. A Channel
borrows its source, so the source and every object it captures must outlive the
Channel.

## Thought Experiments

Choices belong to a Channel, which prevents an id or schema from drifting away
from the bound input:

```cpp fragment
std::array replacement{std::byte{0x2a}};
std::array choices{commands.choice(sequence, replacement)};
auto changed = rund::replay::scenario(session, baseline, choices, simulate);
```

runD validates and freezes all choices before restore or simulation. Duplicate,
missing, ambiguous, invalid, or over-capacity rows return typed failures and do
not invoke the callback. A valid changed result is a successful Scenario;
inspect `matches()` to distinguish equality from a deliberate difference.

## Checkpoint And Resume

A stateful Binding combines the same input-channel factory with one nonzero
checkpoint schema and one borrowed restore codec:

```cpp fragment
auto restore = [&](std::span<const std::byte> bytes) {
  return restore_world(bytes) ? rund::replay::Restore::Restored
                              : rund::replay::Restore::Failed;
};

rund::replay::Binding replay{state_schema, restore};
auto commands = replay.input(identity, source);
auto checkpoint = replay.checkpoint(record, state_bytes);
auto resume = replay.resume(checkpoint);
auto continued = resume.record(session, simulate);
```

Schema, restore, source, and sequence are not repeated at execution sites.
`Binding{}` remains valid for input-only Replay, but checkpoint, advance,
resume, and History publication reject it with `StateSchemaInvalid`.

Checkpoint state is copied once into immutable owned storage. A continuation
Record commits the checkpoint hash into its start identity, so a Record from an
independent boundary cannot be spliced into the chain. `advance` and History
append enforce the same lineage rule.

## Bounded History

`History` owns a fixed-size ring of immutable Record/Checkpoint segments.

```cpp fragment
rund::replay::History history{bounds};
auto appended = replay.append(history, record, state_bytes);
auto segment = history.find(appended.sequence());
```

Retention is bounded simultaneously by segments, bytes, and evidence rows.
Admitted sequences form a contiguous suffix, so `find(sequence)` is `O(1)`
after checked offset arithmetic. Eviction never mutates a previously copied
Segment value.

## Inspect And Persist

- `check(expected, actual)` performs strict equality.
- `diff(expected, actual)` reports typed differences.
- `window(expected, actual, context)` exposes bounded borrowed context around
  the first observation, host-event, input, or trace mismatch.
- Trace context exposes typed `TraceCode` and snapshot `ReasonCode` values;
  `error()` and `snapshot_error()` are stable derived projections.
- `Record::captures()` exposes optional bounded raw ingress for diagnosing
  pre-canonical parser faults; it never becomes simulation input.
- `Record::save` and `Checkpoint::save` stream ordered spans.
- `Record::load` and `Checkpoint::load` enforce explicit `Limits` before
  publishing immutable evidence.

Keep the owning Record, Diff, Window, or Checkpoint alive while using a returned
span or string view.

## Completion

Every fallible Replay value has one outcome authority:

```text
code()       typed replay::Code
ok()/bool    true exactly on success
error()      empty on success, stable text on failure
exit_code()  0 on success, 1 on product failure
```

Application validation mismatches may use process exit `2` only after every
product outcome succeeded.

For a reusable Session, keep the Replay operation result and the shutdown
result distinct while calling `close()` once on every path:

```cpp fragment
const int operation = Replay(session);
const auto closed = session.close();
if (operation != 0 && operation != 2) {
  return operation;
}
return closed ? operation : closed.exit_code();
```

The originating Replay failure wins when both operations fail; a close failure
is returned only after Replay succeeded. This keeps the most specific product
diagnostic, guarantees shutdown, and avoids a caller-authored
`drain(); close();` sequence.

## Executable Journeys

- [Record and Replay](https://github.com/sigee-min/runD/blob/main/package/tests/consumer/example/replay.cpp)
- [Scenario](https://github.com/sigee-min/runD/blob/main/package/tests/consumer/example/scenario.cpp)
- [Checkpoint continuation](https://github.com/sigee-min/runD/blob/main/package/tests/consumer/example/checkpoint.cpp)
- [Bounded History](https://github.com/sigee-min/runD/blob/main/package/tests/consumer/example/history.cpp)
- [Replay telemetry](https://github.com/sigee-min/runD/blob/main/package/tests/consumer/example/telemetry.cpp)

The normative ordering, hash, lineage, storage, and failure laws are in the
[Replay contract](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/replay.md). Session lifecycle is in
[Run](./Run.md), and diagnostics are in [Telemetry](./Telemetry.md).
