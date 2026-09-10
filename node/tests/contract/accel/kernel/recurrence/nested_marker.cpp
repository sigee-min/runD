#include "local.hpp"

namespace node_accel_contract {
namespace {

[[nodiscard]] bool NestedMarkerContract() {
  constexpr std::size_t Outer = 2u;
  constexpr std::size_t Inner = 2u;
  constexpr std::size_t Count = Outer + Inner + 3u;
  Fixture fixture{ComputeApi::Metal, ComputeScalar::Lane32};
  std::array<BackendWindow, Count> windows{};
  std::array<BackendBatchEntry, Count> entries{};
  std::array<std::uint8_t, Count> barriers{};
  barriers.fill(1u);
  for (std::size_t index = 0u; index < Count; ++index) {
    const bool seed = index < Outer;
    const bool action = index >= Outer && index < Outer + Inner;
    const std::uint32_t iteration = static_cast<std::uint32_t>(
        seed ? index : (action ? index - Outer : index - Outer - Inner));
    const std::uint32_t bound =
        seed ? static_cast<std::uint32_t>(Outer)
             : (action ? static_cast<std::uint32_t>(Inner) : 3u);
    windows[index] = BackendWindow{
        .maximum = 8u,
        .tile = 4u,
        .state = 3u,
        .outer_iteration = seed ? iteration : 0u,
        .outer_bound = static_cast<std::uint32_t>(Outer),
        .inner_iteration = action ? iteration : 0u,
        .inner_bound = static_cast<std::uint32_t>(Inner),
        .inner_advance = action ? 1u : 0u,
        .route = seed || action ? 0u : iteration,
        .phase = seed ? BackendWindowPhase::NestedSeed
                      : (action ? BackendWindowPhase::NestedAction
                                : BackendWindowPhase::NestedFold),
    };
    if (action) {
      entries[index] = fixture.entries[index - Outer];
    }
    entries[index].recurrence = BackendRecurrence{
        .logical_step = 7u,
        .iteration = iteration,
        .bound = bound,
        .window = &windows[index],
    };
  }
  const auto build = [&] {
    NestedTemplateGeometry geometry{};
    if (!ProveNestedTemplateGeometry(
            std::span<const BackendBatchEntry>{entries.data(), entries.size()},
            0u, geometry)) {
      return MapRecurrence{};
    }
    return BuildNestedMapRecurrence(
        std::span<const BackendBatchEntry>{
            entries.data() + geometry.action_first(), geometry.inner_bound()},
        std::span<const std::uint8_t>{barriers.data() + geometry.action_first(),
                                      geometry.inner_bound()},
        geometry);
  };
  const MapRecurrence ready = build();
  if (!ready.ready() || ready.iterations != Inner) {
    return false;
  }
  NestedTemplateGeometry token{};
  if (!ProveNestedTemplateGeometry(
          std::span<const BackendBatchEntry>{entries.data(), entries.size()},
          0u, token)) {
    return false;
  }
  const std::array<BackendBatchEntry, Inner> detached_actions{
      entries[Outer], entries[Outer + 1u]};
  if (BuildNestedMapRecurrence(
          detached_actions,
          std::span<const std::uint8_t>{barriers.data() + Outer, Inner}, token)
          .state != MapRecurrenceState::Ineligible) {
    return false;
  }

  barriers[Outer + 1u] = 0u;
  if (build().state != MapRecurrenceState::Invalid) {
    return false;
  }
  barriers[Outer + 1u] = 1u;
  windows[Outer + 1u].inner_iteration = 0u;
  if (build().state != MapRecurrenceState::Ineligible) {
    return false;
  }
  windows[Outer + 1u].inner_iteration = 1u;
  entries[Outer].recurrence.writes_each_iteration = true;
  if (build().state != MapRecurrenceState::Ineligible) {
    return false;
  }
  entries[Outer].recurrence.writes_each_iteration = false;

  constexpr std::size_t OneCount = Outer + 1u + 3u;
  std::array<BackendBatchEntry, OneCount> one_entries{};
  std::array<std::uint8_t, OneCount> one_barriers{};
  one_barriers.fill(1u);
  for (BackendWindow &window : windows) {
    window.inner_bound = 1u;
  }
  one_entries[0u] = entries[0u];
  one_entries[1u] = entries[1u];
  one_entries[2u] = entries[Outer];
  one_entries[2u].recurrence.bound = 1u;
  one_entries[3u] = entries[Outer + Inner];
  one_entries[4u] = entries[Outer + Inner + 1u];
  one_entries[5u] = entries[Outer + Inner + 2u];
  NestedTemplateGeometry one_geometry{};
  if (!ProveNestedTemplateGeometry(
          std::span<const BackendBatchEntry>{one_entries.data(),
                                             one_entries.size()},
          0u, one_geometry) ||
      BuildNestedMapRecurrence(
          std::span<const BackendBatchEntry>{one_entries.data() +
                                                 one_geometry.action_first(),
                                             one_geometry.inner_bound()},
          std::span<const std::uint8_t>{one_barriers.data() +
                                            one_geometry.action_first(),
                                        one_geometry.inner_bound()},
          one_geometry)
              .state != MapRecurrenceState::Ineligible) {
    return false;
  }
  for (BackendWindow &window : windows) {
    window.inner_bound = static_cast<std::uint32_t>(Inner);
  }

  fixture.occurrences[0u].step.artifact.metadata.read_routes.push_back(
      rund::kernel::ReadRoute{.source = 0u, .index = 1u, .count = 4u});
  if (build().state != MapRecurrenceState::Ineligible) {
    return false;
  }
  return true;
}

} // namespace

bool MapRecurrenceNestedMarkerContract() {
  Fixture ordinary{ComputeApi::Metal, ComputeScalar::Lane32};
  ordinary.entries[0].recurrence.bound = 1u;
  ordinary.entries[1].recurrence.bound = 1u;
  if (BuildMapRecurrence(ordinary.entries, ordinary.barriers).state !=
      MapRecurrenceState::Ineligible) {
    return false;
  }
  return NestedMarkerContract();
}

} // namespace node_accel_contract
