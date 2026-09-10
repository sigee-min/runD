#pragma once

#include "../../../accel/kernel/nested.hpp"
#include <rund/compute/pipeline/profile.hpp>

namespace rund::compute::detail {

// A nested window keeps one compact table of reusable routes.  Route entries
// are not execution occurrences: seed has one ordinal-specific route per
// outer window, action has one route per inner parity occurrence, and fold has
// the three seed/first/second outer-state transitions.  The warm executor
// interprets the two independent bounds without materializing their product.
enum class PipelineRoute : std::uint8_t {
  Ordinary,
  NestedSeed,
  NestedAction,
  NestedFold,
};

[[nodiscard]] inline constexpr PipelineRoute
pipeline_route(const node::accel::detail::NestedTemplatePhase phase) noexcept {
  switch (phase) {
  case node::accel::detail::NestedTemplatePhase::Seed:
    return PipelineRoute::NestedSeed;
  case node::accel::detail::NestedTemplatePhase::Action:
    return PipelineRoute::NestedAction;
  case node::accel::detail::NestedTemplatePhase::Fold:
    return PipelineRoute::NestedFold;
  }
  return PipelineRoute::Ordinary;
}

// Sole Compute-owned projection from one frozen route to its public nested
// phase. Planning diagnostics, profiles, and synchronous/asynchronous failure
// paths consume this table; outer/inner coordinates remain contextual because
// compact Action/Fold routes do not identify a physical outer occurrence.
[[nodiscard]] inline constexpr PipelineNestedPhase
pipeline_nested_phase(const PipelineRoute route) noexcept {
  switch (route) {
  case PipelineRoute::NestedSeed:
    return PipelineNestedPhase::Seed;
  case PipelineRoute::NestedAction:
    return PipelineNestedPhase::Action;
  case PipelineRoute::NestedFold:
    return PipelineNestedPhase::Fold;
  case PipelineRoute::Ordinary:
    return PipelineNestedPhase::None;
  }
  return PipelineNestedPhase::None;
}

} // namespace rund::compute::detail
