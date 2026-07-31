#include <rund/replay.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

int main() {
  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.workers = 1u});
  if (!opened) {
    return opened.exit_code();
  }

  constexpr std::uint64_t sequence = 1u;
  std::byte expected = std::byte{7};
  std::uint32_t source_calls = 0u;
  bool observed = false;
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++source_calls;
    const std::array bytes{std::byte{7}};
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
    const rund::replay::Record baseline =
        rund::replay::record(session, simulate);
    if (!baseline) {
      return baseline.exit_code();
    }
    if (!observed || source_calls != 1u) {
      return 2;
    }

    const std::array replacement{std::byte{9}};
    const std::array choices{commands.choice(sequence, replacement)};
    expected = replacement[0];
    observed = false;
    const rund::replay::Scenario explored =
        rund::replay::scenario(session, baseline, choices, simulate);
    if (!explored) {
      return explored.exit_code();
    }
    if (!explored.callback_ran() || explored.matches() ||
        !explored.actual().has_value() || !explored.diff().has_value() ||
        !observed || source_calls != 1u) {
      return 2;
    }
    return 0;
  }();

  const auto closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  return closed ? operation : closed.exit_code();
}
