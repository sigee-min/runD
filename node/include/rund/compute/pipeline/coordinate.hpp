#pragma once

#include <cstdint>

namespace rund::compute {

enum class PipelineNestedPhase : std::uint8_t {
  None,
  Seed,
  Action,
  Fold,
};

// Nested coordinates have one canonical public shape. Ordinary work has no
// nested coordinates, Seed/Fold identify only their outer window, and Action
// identifies both the outer window and inner iteration. Keeping this law next
// to the phase prevents preparation and runtime failure paths from inventing
// an inner coordinate for Seed/Fold or leaking default zeroes for ordinary
// work.
[[nodiscard]] constexpr bool
pipeline_nested_phase_valid(const PipelineNestedPhase phase) noexcept {
  switch (phase) {
  case PipelineNestedPhase::None:
  case PipelineNestedPhase::Seed:
  case PipelineNestedPhase::Action:
  case PipelineNestedPhase::Fold:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool
pipeline_nested_phase_has_outer(const PipelineNestedPhase phase) noexcept {
  return phase == PipelineNestedPhase::Seed ||
         phase == PipelineNestedPhase::Action ||
         phase == PipelineNestedPhase::Fold;
}

[[nodiscard]] constexpr bool
pipeline_nested_phase_has_inner(const PipelineNestedPhase phase) noexcept {
  return phase == PipelineNestedPhase::Action;
}

[[nodiscard]] constexpr bool valid_pipeline_nested_coordinate(
    const PipelineNestedPhase phase, const bool outer_known,
    const bool inner_known) noexcept {
  return pipeline_nested_phase_valid(phase) &&
         outer_known == pipeline_nested_phase_has_outer(phase) &&
         inner_known == pipeline_nested_phase_has_inner(phase);
}

} // namespace rund::compute
