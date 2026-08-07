#pragma once

#include <array>
#include <cstdint>

namespace rund::compute {

enum class PipelineNestedPhase : std::uint8_t {
  None = 0u,
  Seed = 1u,
  Action = 2u,
  Fold = 3u,
};

namespace detail {

// This table is the sole owner of the admitted public phase set, its
// coordinate shape, and the stable source key used by generated device code.
// Consumers must look up a row rather than recreate a four-way validity list.
struct PipelineNestedPhaseContract final {
  PipelineNestedPhase phase;
  const char *source_key;
  bool has_outer;
  bool has_inner;
};

inline constexpr std::array<PipelineNestedPhaseContract, 4u>
    PipelineNestedPhaseContracts{{
        {PipelineNestedPhase::None, "none", false, false},
        {PipelineNestedPhase::Seed, "seed", true, false},
        {PipelineNestedPhase::Action, "action", true, true},
        {PipelineNestedPhase::Fold, "fold", true, false},
    }};

[[nodiscard]] constexpr const PipelineNestedPhaseContract *
pipeline_nested_phase_contract(const PipelineNestedPhase phase) noexcept {
  for (const PipelineNestedPhaseContract &contract :
       PipelineNestedPhaseContracts) {
    if (contract.phase == phase) {
      return &contract;
    }
  }
  return nullptr;
}

} // namespace detail

// Nested coordinates have one canonical public shape. Ordinary work has no
// nested coordinates, Seed/Fold identify only their outer window, and Action
// identifies both the outer window and inner iteration. Keeping this law next
// to the phase prevents preparation and runtime failure paths from inventing
// an inner coordinate for Seed/Fold or leaking default zeroes for ordinary
// work.
[[nodiscard]] constexpr bool
pipeline_nested_phase_valid(const PipelineNestedPhase phase) noexcept {
  return detail::pipeline_nested_phase_contract(phase) != nullptr;
}

[[nodiscard]] constexpr bool
pipeline_nested_phase_has_outer(const PipelineNestedPhase phase) noexcept {
  const detail::PipelineNestedPhaseContract *const contract =
      detail::pipeline_nested_phase_contract(phase);
  return contract != nullptr && contract->has_outer;
}

[[nodiscard]] constexpr bool
pipeline_nested_phase_has_inner(const PipelineNestedPhase phase) noexcept {
  const detail::PipelineNestedPhaseContract *const contract =
      detail::pipeline_nested_phase_contract(phase);
  return contract != nullptr && contract->has_inner;
}

[[nodiscard]] constexpr bool
valid_pipeline_nested_coordinate(const PipelineNestedPhase phase,
                                 const bool outer_known,
                                 const bool inner_known) noexcept {
  const detail::PipelineNestedPhaseContract *const contract =
      detail::pipeline_nested_phase_contract(phase);
  return contract != nullptr && outer_known == contract->has_outer &&
         inner_known == contract->has_inner;
}

} // namespace rund::compute
