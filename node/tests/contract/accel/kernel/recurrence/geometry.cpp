#include "local.hpp"

namespace node_accel_contract {
namespace {

[[nodiscard]] bool NestedTemplateShapeContract() {
  struct ShapeCase final {
    std::uint32_t outer{};
    std::uint32_t inner{};
    std::uint64_t compact{};
    std::uint64_t retained{};
    std::uint64_t authored{};
    std::uint64_t action_occurrences{};
    std::array<std::uint64_t, 3u> fold_occurrences{};
    bool action_candidate{};
  };
  constexpr std::array<ShapeCase, 12u> cases{{
      {1u, 0u, 4u, 4u, 2u, 0u, {1u, 0u, 0u}, false},
      {1u, 1u, 5u, 5u, 3u, 1u, {1u, 0u, 0u}, false},
      {1u, 2u, 6u, 6u, 4u, 2u, {1u, 0u, 0u}, true},
      {2u, 0u, 5u, 5u, 4u, 0u, {1u, 1u, 0u}, false},
      {2u, 1u, 6u, 6u, 6u, 2u, {1u, 1u, 0u}, false},
      {2u, 2u, 7u, 7u, 8u, 4u, {1u, 1u, 0u}, true},
      {3u, 0u, 6u, 6u, 6u, 0u, {1u, 1u, 1u}, false},
      {3u, 1u, 7u, 7u, 9u, 3u, {1u, 1u, 1u}, false},
      {3u, 2u, 8u, 8u, 12u, 6u, {1u, 1u, 1u}, true},
      {4u, 0u, 7u, 7u, 8u, 0u, {1u, 2u, 1u}, false},
      {4u, 1u, 8u, 8u, 12u, 4u, {1u, 2u, 1u}, false},
      {4u, 2u, 9u, 9u, 16u, 8u, {1u, 2u, 1u}, true},
  }};
  constexpr std::size_t first = 5u;
  constexpr std::array<std::uint32_t, 4u> fold_routes{0u, 1u, 2u, 1u};
  for (const ShapeCase &expected : cases) {
    NestedTemplateShape shape{};
    if (!ProveNestedTemplateShape(first, expected.outer, 1u, expected.inner,
                                  shape) ||
        !shape.valid() || shape.first() != first ||
        shape.action_first() != first + expected.outer ||
        shape.fold_first() != first + expected.outer + expected.inner ||
        shape.end() != first + expected.compact ||
        shape.outer_bound() != expected.outer ||
        shape.inner_bound() != expected.inner ||
        shape.compact_entry_count() != expected.compact ||
        shape.retained_entry_count() != expected.retained ||
        shape.authored_occurrence_count() != expected.authored ||
        shape.authored_seed_occurrence_count() != expected.outer ||
        shape.authored_action_occurrence_count() !=
            expected.action_occurrences ||
        shape.authored_fold_occurrence_count() != expected.outer ||
        shape.transduced_occurrence_count() != expected.outer * 3u ||
        shape.action_group_candidate() != expected.action_candidate) {
      return false;
    }
    for (std::uint32_t outer = 0u; outer < expected.outer; ++outer) {
      NestedTemplateRouteProjection route{};
      if (!shape.project(shape.seed_first() + outer, route) ||
          route.phase != NestedTemplatePhase::Seed ||
          route.occurrence_count != 1u || route.iteration != outer ||
          route.bound != expected.outer || route.outer_iteration != outer ||
          route.outer_bound != expected.outer ||
          route.inner_bound != expected.inner) {
        return false;
      }
      std::uint32_t fold_route = 0u;
      if (!shape.fold_route_for_outer(outer, fold_route) ||
          fold_route != fold_routes[outer]) {
        return false;
      }
    }
    for (std::uint32_t inner = 0u; inner < expected.inner; ++inner) {
      NestedTemplateRouteProjection route{};
      if (!shape.project(shape.action_first() + inner, route) ||
          route.phase != NestedTemplatePhase::Action ||
          route.occurrence_count != expected.outer ||
          route.iteration != inner || route.bound != expected.inner ||
          route.inner_iteration != inner || route.inner_advance != 1u) {
        return false;
      }
    }
    for (std::uint32_t fold = 0u; fold < 3u; ++fold) {
      NestedTemplateRouteProjection route{};
      if (!shape.project(shape.fold_first() + fold, route) ||
          route.phase != NestedTemplatePhase::Fold ||
          route.occurrence_count != expected.fold_occurrences[fold] ||
          route.iteration != fold || route.bound != 3u || route.route != fold) {
        return false;
      }
    }
  }

  NestedTemplateShape tail{};
  if (!ProveNestedTemplateShape(7u, 5u, 2u, 1u, tail) ||
      tail.outer_bound() != 3u || tail.seed_first() != 7u ||
      tail.action_first() != 10u || tail.fold_first() != 11u ||
      tail.end() != 14u || tail.authored_occurrence_count() != 9u) {
    return false;
  }

  NestedTemplateShape parity{};
  if (!ProveNestedTemplateShape(10u, 2u, 1u, 4u, parity) ||
      parity.compact_entry_count() != 9u ||
      parity.retained_entry_count() != 7u) {
    return false;
  }
  constexpr std::array<std::size_t, 4u> expected_owners{12u, 13u, 12u, 13u};
  for (std::size_t offset = 0u; offset < expected_owners.size(); ++offset) {
    std::size_t owner = 0u;
    if (!parity.retained_owner(parity.action_first() + offset, owner) ||
        owner != expected_owners[offset]) {
      return false;
    }
  }
  NestedTemplateShape invalid{};
  return !ProveNestedTemplateShape(0u, 0u, 1u, 1u, invalid) &&
         !ProveNestedTemplateShape(0u, 1u, 0u, 1u, invalid) &&
         !ProveNestedTemplateShape(0u, 1u, 2u, 1u, invalid) &&
         !ProveNestedTemplateShape(std::numeric_limits<std::size_t>::max() - 2u,
                                   5u, 2u, 1u, invalid);
}

[[nodiscard]] bool NestedTemplateGeometryContract() {
  constexpr std::size_t Outer = 2u;
  constexpr std::size_t Inner = 2u;
  constexpr std::size_t Count = Outer + Inner + 3u;
  std::array<BackendWindow, Count> windows{};
  std::array<BackendRecurrence, Count> recurrences{};
  for (std::size_t index = 0u; index < Count; ++index) {
    BackendWindowPhase phase = BackendWindowPhase::NestedFold;
    std::uint32_t iteration = static_cast<std::uint32_t>(index - Outer - Inner);
    std::uint32_t bound = 3u;
    std::uint32_t outer = 0u;
    std::uint32_t inner = 0u;
    std::uint32_t route = iteration;
    std::uint32_t advance = 0u;
    if (index < Outer) {
      phase = BackendWindowPhase::NestedSeed;
      iteration = static_cast<std::uint32_t>(index);
      bound = static_cast<std::uint32_t>(Outer);
      outer = iteration;
      route = 0u;
    } else if (index < Outer + Inner) {
      phase = BackendWindowPhase::NestedAction;
      iteration = static_cast<std::uint32_t>(index - Outer);
      bound = static_cast<std::uint32_t>(Inner);
      inner = iteration;
      route = 0u;
      advance = 1u;
    }
    windows[index] = BackendWindow{
        .maximum = 8u,
        .tile = 4u,
        .state = 3u,
        .outer_iteration = outer,
        .outer_bound = static_cast<std::uint32_t>(Outer),
        .inner_iteration = inner,
        .inner_bound = static_cast<std::uint32_t>(Inner),
        .inner_advance = advance,
        .route = route,
        .phase = phase,
    };
    recurrences[index] = BackendRecurrence{
        .logical_step = 7u,
        .iteration = iteration,
        .bound = bound,
        .window = &windows[index],
    };
  }
  const auto batch = [&] {
    std::array<BackendBatchEntry, Count> entries{};
    for (std::size_t index = 0u; index < Count; ++index) {
      entries[index] = BackendBatchEntry{
          .recurrence = recurrences[index],
          .template_index = static_cast<std::uint32_t>(index),
      };
    }
    return entries;
  };
  const auto proves = [&](const bool expected) {
    NestedTemplateGeometry recurrence_geometry{};
    NestedTemplateGeometry batch_geometry{};
    const auto entries = batch();
    const bool recurrence_ok = ProveNestedTemplateGeometry(
        std::span<const BackendRecurrence>{recurrences.data(),
                                           recurrences.size()},
        0u, recurrence_geometry);
    const bool batch_ok = ProveNestedTemplateGeometry(
        std::span<const BackendBatchEntry>{entries.data(), entries.size()}, 0u,
        batch_geometry);
    if (recurrence_ok != expected || batch_ok != expected) {
      return false;
    }
    return !expected ||
           (recurrence_geometry.first() == 0u &&
            recurrence_geometry.action_first() == Outer &&
            recurrence_geometry.fold_first() == Outer + Inner &&
            recurrence_geometry.end() == Count &&
            recurrence_geometry.outer_bound() == Outer &&
            recurrence_geometry.inner_bound() == Inner &&
            batch_geometry.action_first() ==
                recurrence_geometry.action_first() &&
            batch_geometry.fold_first() == recurrence_geometry.fold_first() &&
            batch_geometry.end() == recurrence_geometry.end());
  };
  if (!proves(true)) {
    return false;
  }

  for (BackendWindow &window : windows) {
    window.maximum = 12u;
  }
  if (!proves(false)) {
    return false;
  }
  for (BackendWindow &window : windows) {
    window.maximum = 8u;
  }
  windows[Outer].phase = BackendWindowPhase::NestedFold;
  if (!proves(false)) {
    return false;
  }
  windows[Outer].phase = BackendWindowPhase::NestedAction;
  windows[0u].outer_iteration = 1u;
  if (!proves(false)) {
    return false;
  }
  windows[0u].outer_iteration = 0u;
  windows[Outer].inner_advance = 0u;
  if (!proves(false)) {
    return false;
  }
  windows[Outer].inner_advance = 1u;
  windows[Outer + Inner].route = 1u;
  if (!proves(false)) {
    return false;
  }
  windows[Outer + Inner].route = 0u;
  recurrences[Outer + 1u].iteration = 0u;
  if (!proves(false)) {
    return false;
  }
  recurrences[Outer + 1u].iteration = 1u;
  recurrences[Outer + 1u].bound = 1u;
  if (!proves(false)) {
    return false;
  }
  recurrences[Outer + 1u].bound = static_cast<std::uint32_t>(Inner);
  recurrences[Outer + Inner].logical_step = 8u;
  if (!proves(false)) {
    return false;
  }
  recurrences[Outer + Inner].logical_step = 7u;
  recurrences[Outer].writes_each_iteration = true;
  if (!proves(false)) {
    return false;
  }
  recurrences[Outer].writes_each_iteration = false;
  windows[Outer + 1u].count.source.stride_bytes = 8u;
  if (!proves(false)) {
    return false;
  }
  windows[Outer + 1u].count.source.stride_bytes = 0u;
  windows[Outer + 1u].count.handle = std::make_shared<std::uint32_t>(1u);
  if (!proves(false)) {
    return false;
  }
  windows[Outer + 1u].count.handle.reset();

  for (BackendWindow &window : windows) {
    window.has_terminal = true;
  }
  if (!proves(true)) {
    return false;
  }
  windows.back().terminal[1u].source.id = 1u;
  if (!proves(false)) {
    return false;
  }
  windows.back().terminal[1u].source.id = 0u;

  std::array<std::uint8_t, Count> barriers{};
  barriers.fill(1u);
  auto entries = batch();
  const auto publications =
      std::span<const rund::node::accel::detail::BackendPublish>{};
  const auto terminal_aggregate =
      BuildNestedAggregate(entries, barriers, publications, 0u);
  if (terminal_aggregate.state != NestedAggregateState::Ineligible ||
      std::string_view{terminal_aggregate.reason} !=
          "compute_pipeline_nested_aggregate_shape_ineligible") {
    return false;
  }
  for (BackendWindow &window : windows) {
    window.has_terminal = false;
  }
  entries = batch();
  BackendRun aggregate_run{};
  for (BackendBatchEntry &entry : entries) {
    entry.run = &aggregate_run;
  }
  const auto base_aggregate =
      BuildNestedAggregate(entries, barriers, publications, 0u);
  if (base_aggregate.state != NestedAggregateState::Ineligible ||
      std::string_view{base_aggregate.reason} !=
          "compute_pipeline_nested_aggregate_seed_ineligible") {
    return false;
  }
  barriers[1u] = 0u;
  const auto barrier_aggregate =
      BuildNestedAggregate(entries, barriers, publications, 0u);
  if (barrier_aggregate.state != NestedAggregateState::Ineligible ||
      std::string_view{barrier_aggregate.reason} !=
          "compute_pipeline_nested_aggregate_shape_ineligible") {
    return false;
  }

  constexpr std::size_t ZeroCount = Outer + 3u;
  std::array<BackendWindow, ZeroCount> zero_windows{};
  std::array<BackendRecurrence, ZeroCount> zero_recurrences{};
  std::array<BackendBatchEntry, ZeroCount> zero_entries{};
  for (std::size_t index = 0u; index < ZeroCount; ++index) {
    const bool seed = index < Outer;
    const std::uint32_t iteration =
        static_cast<std::uint32_t>(seed ? index : index - Outer);
    zero_windows[index] = BackendWindow{
        .maximum = 8u,
        .tile = 4u,
        .state = 3u,
        .outer_iteration = seed ? iteration : 0u,
        .outer_bound = static_cast<std::uint32_t>(Outer),
        .inner_bound = 0u,
        .route = seed ? 0u : iteration,
        .phase = seed ? BackendWindowPhase::NestedSeed
                      : BackendWindowPhase::NestedFold,
    };
    zero_recurrences[index] = BackendRecurrence{
        .logical_step = 9u,
        .iteration = iteration,
        .bound = seed ? static_cast<std::uint32_t>(Outer) : 3u,
        .window = &zero_windows[index],
    };
    zero_entries[index] = BackendBatchEntry{
        .recurrence = zero_recurrences[index],
        .template_index = static_cast<std::uint32_t>(index),
    };
  }
  NestedTemplateGeometry zero_geometry{};
  NestedTemplateGeometry zero_batch_geometry{};
  std::array<std::uint8_t, ZeroCount> zero_barriers{};
  zero_barriers.fill(1u);
  const auto zero_aggregate =
      BuildNestedAggregate(zero_entries, zero_barriers, publications, 0u);
  return ProveNestedTemplateGeometry(
             std::span<const BackendRecurrence>{zero_recurrences.data(),
                                                zero_recurrences.size()},
             0u, zero_geometry) &&
         ProveNestedTemplateGeometry(
             std::span<const BackendBatchEntry>{zero_entries.data(),
                                                zero_entries.size()},
             0u, zero_batch_geometry) &&
         zero_geometry.inner_bound() == 0u &&
         BuildNestedMapRecurrence(
             std::span<const BackendBatchEntry>{zero_entries.data() + Outer,
                                                0u},
             std::span<const std::uint8_t>{zero_barriers.data() + Outer, 0u},
             zero_batch_geometry)
                 .state == MapRecurrenceState::Ineligible &&
         zero_aggregate.state == NestedAggregateState::Ineligible &&
         std::string_view{zero_aggregate.reason} ==
             "compute_pipeline_nested_aggregate_shape_ineligible";
}

} // namespace

bool MapRecurrenceGeometryContract() {
  return NestedTemplateShapeContract() &&
         NestedTemplateGeometryContract();
}

} // namespace node_accel_contract
