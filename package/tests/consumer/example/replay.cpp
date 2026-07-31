#include <rund/replay.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

int Replay(rund::Session &session) {
  std::byte produced = std::byte{7};
  std::uint32_t source_calls = 0u;
  bool simulation_ok = false;
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++source_calls;
    const std::array bytes{produced};
    (void)writer.append(bytes);
    return 1u;
  };
  const auto commands =
      replay.input(rund::replay::Input{.id = 1u, .schema = 1u}, source);
  auto simulate = [&](rund::replay::Context &context) {
    const auto value = commands.read(context);
    simulation_ok = value && value.sequence() == 1u && value.size() == 1u &&
                    value.bytes()[0] == std::byte{7};
  };

  const auto recorded = rund::replay::record(session, simulate);
  if (!recorded) {
    return recorded.exit_code();
  }
  produced = std::byte{255};
  simulation_ok = false;
  const auto replayed = rund::replay::run(session, recorded, simulate);
  if (!replayed) {
    return replayed.exit_code();
  }
  return simulation_ok && source_calls == 1u ? 0 : 2;
}

} // namespace

int main() {
  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.workers = 1u});
  if (!opened) {
    return opened.exit_code();
  }

  const int operation = Replay(session);
  const auto closed = session.close();
  if (operation != 0 && operation != 2) {
    return operation;
  }
  return closed ? operation : closed.exit_code();
}
