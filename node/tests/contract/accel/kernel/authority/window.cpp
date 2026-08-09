#include "src/accel/kernel/backend/run.hpp"

#include <cstddef>

#include "window.hpp"

namespace node_accel_contract {

using BackendWindow = rund::node::accel::detail::BackendWindow;

[[nodiscard]] bool BackendWindowDefaultsFailClosed() {
  return BackendWindow{}.outer_bound == 0u;
}

[[nodiscard]] bool BackendWindowOccurrenceShapeHasOneAuthority() {
  using rund::node::accel::detail::BackendWindowPhase;
  const BackendWindow ordinary{
      .maximum = 8u,
      .tile = 4u,
      .outer_iteration = 1u,
      .outer_bound = 2u,
  };
  if (!ordinary.valid_occurrence(false) || ordinary.valid_occurrence(true)) {
    return false;
  }
  for (std::size_t invalid = 0u; invalid < 5u; ++invalid) {
    BackendWindow window = ordinary;
    switch (invalid) {
    case 0u:
      window.maximum = 0u;
      break;
    case 1u:
      window.tile = 0u;
      break;
    case 2u:
      window.tile = window.maximum + 1u;
      break;
    case 3u:
      window.outer_bound = 0u;
      break;
    case 4u:
      window.outer_iteration = window.outer_bound;
      break;
    }
    if (window.valid_occurrence(false)) {
      return false;
    }
  }

  BackendWindow seed = ordinary;
  seed.phase = BackendWindowPhase::NestedSeed;
  if (!seed.valid_occurrence(false) || seed.valid_occurrence(true)) {
    return false;
  }
  seed.inner_advance = 1u;
  if (seed.valid_occurrence(false)) {
    return false;
  }
  seed.inner_advance = 0u;
  seed.route = 1u;
  if (seed.valid_occurrence(false)) {
    return false;
  }

  BackendWindow action = ordinary;
  action.phase = BackendWindowPhase::NestedAction;
  action.inner_iteration = 2u;
  action.inner_bound = 3u;
  action.inner_advance = 1u;
  if (!action.valid_occurrence(false) || action.valid_occurrence(true)) {
    return false;
  }
  action.inner_advance = 0u;
  if (!action.valid_occurrence(true) || action.valid_occurrence(false)) {
    return false;
  }
  action.inner_advance = 1u;
  action.inner_iteration = action.inner_bound;
  if (action.valid_occurrence(false)) {
    return false;
  }
  action.inner_iteration = 2u;
  action.route = 1u;
  if (action.valid_occurrence(false)) {
    return false;
  }
  action.route = 0u;
  action.inner_iteration = 0u;
  action.inner_bound = 0u;
  if (action.valid_occurrence(false)) {
    return false;
  }

  BackendWindow fold = ordinary;
  fold.phase = BackendWindowPhase::NestedFold;
  fold.inner_bound = 3u;
  if (!fold.valid_occurrence(false) || fold.valid_occurrence(true)) {
    return false;
  }
  fold.route = 2u;
  fold.inner_advance = fold.inner_bound;
  if (!fold.valid_occurrence(false)) {
    return false;
  }
  fold.route = 3u;
  if (fold.valid_occurrence(false)) {
    return false;
  }
  fold.route = 2u;
  fold.inner_advance = 1u;
  if (fold.valid_occurrence(false)) {
    return false;
  }

  BackendWindow invalid_phase = ordinary;
  invalid_phase.phase = static_cast<BackendWindowPhase>(0xffu);
  return !invalid_phase.valid_occurrence(false);
}

} // namespace node_accel_contract
