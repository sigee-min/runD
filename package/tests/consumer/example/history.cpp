#include <rund/replay.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

int main() {
  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.workers = 1u});
  if (!opened) {
    return opened.exit_code();
  }

  constexpr std::uint64_t state_schema = 7u;
  std::uint64_t sequence = 1u;
  std::byte produced = std::byte{7};
  std::byte expected = produced;
  std::uint32_t source_calls = 0u;
  bool observed = false;
  auto restore = [](const std::span<const std::byte> bytes) {
    return bytes.size() == 1u && bytes[0] == std::byte{42}
               ? rund::replay::Restore::Restored
               : rund::replay::Restore::Failed;
  };
  rund::replay::Binding replay{state_schema, restore};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++source_calls;
    const std::array bytes{produced};
    (void)writer.append(bytes);
    return sequence;
  };
  const auto commands =
      replay.input(rund::replay::Input{.id = 1u, .schema = 1u}, source);
  auto simulate = [&](rund::replay::Context &context) {
    const auto value = commands.read(context);
    observed = value && value.sequence() == sequence && value.size() == 1u &&
               value.bytes()[0] == expected;
  };

  const int operation = [&] {
    const rund::replay::Record first = rund::replay::record(session, simulate);
    if (!first) {
      return first.exit_code();
    }
    if (!observed || source_calls != 1u) {
      return 2;
    }

    const std::array first_state{std::byte{42}};
    rund::replay::History history{rund::replay::Retention{
        .max_segments = 1u,
        .max_bytes = 1024u * 1024u * 1024u,
        .max_events = 1024u * 1024u,
    }};
    if (!history) {
      return history.exit_code();
    }
    const rund::replay::Append added =
        replay.append(history, first, first_state);
    if (!added) {
      return added.exit_code();
    }
    const auto previous = history.find(added.sequence());
    if (!previous.has_value()) {
      return 2;
    }

    const auto resume = replay.resume(previous->checkpoint());
    if (!resume) {
      return resume.exit_code();
    }
    sequence = 2u;
    produced = std::byte{9};
    expected = produced;
    observed = false;
    const rund::replay::Record second = resume.record(session, simulate);
    if (!second) {
      return second.exit_code();
    }
    if (!observed || source_calls != 2u) {
      return 2;
    }

    const std::array second_state{std::byte{43}};
    const rund::replay::Append replaced =
        replay.append(history, second, second_state);
    if (!replaced) {
      return replaced.exit_code();
    }
    const auto current = history.find(replaced.sequence());
    if (replaced.evicted_segments() != 1u ||
        history.find(added.sequence()).has_value() || !current.has_value()) {
      return 2;
    }

    observed = false;
    const rund::replay::Check replayed =
        resume.run(session, current->record(), simulate);
    if (!replayed) {
      return replayed.exit_code();
    }
    return observed && source_calls == 2u ? 0 : 2;
  }();

  const auto closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  return closed ? operation : closed.exit_code();
}
