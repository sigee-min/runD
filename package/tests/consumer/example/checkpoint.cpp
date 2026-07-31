#include <rund/replay.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

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
  bool simulation_ok = false;
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
    simulation_ok = value && value.sequence() == sequence &&
                    value.size() == 1u && value.bytes()[0] == expected;
  };

  const int operation = [&] {
    const auto initial = rund::replay::record(session, simulate);
    if (!initial)
      return initial.exit_code();
    if (!simulation_ok || source_calls != 1u)
      return 2;
    const std::array snapshot{std::byte{42}};
    const auto captured = replay.checkpoint(initial, snapshot);
    if (!captured)
      return captured.exit_code();
    std::vector<std::byte> persisted{};
    const rund::replay::Save write =
        captured.save([&persisted](const std::span<const std::byte> bytes) {
          persisted.insert(persisted.end(), bytes.begin(), bytes.end());
          return true;
        });
    if (!write)
      return write.exit_code();
    const auto loaded = rund::replay::Checkpoint::load(
        persisted, rund::replay::Limits{.max_state_bytes = snapshot.size()});
    if (!loaded)
      return loaded.exit_code();

    const auto &checkpoint = *loaded;
    const auto resume = replay.resume(checkpoint);
    if (!resume)
      return resume.exit_code();
    sequence = 2u;
    produced = std::byte{9};
    expected = produced;
    const auto continued = resume.record(session, simulate);
    if (!continued)
      return continued.exit_code();

    produced = std::byte{255};
    simulation_ok = false;
    const auto checked = resume.run(session, continued, simulate);
    if (!checked)
      return checked.exit_code();
    return simulation_ok && source_calls == 2u ? 0 : 2;
  }();
  const auto closed = session.close();
  if (operation != 0 && operation != 2)
    return operation;
  if (!closed)
    return closed.exit_code();
  return operation;
}
