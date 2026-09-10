#pragma once

#include "base.hpp"
#include "../plan/state.hpp"
#include "route.hpp"

#include "../../../accel/kernel/nested.hpp"

#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/pipeline/window.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct BufferState;

struct DeviceState;

struct ProgramState;

struct StateSnapshotState;

struct SnapshotStorageState;

struct PipelinePublicationState;

struct PipelineBuildWindowControlOrdinal final {
  static constexpr std::uint32_t unassigned =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t value{unassigned};
};

// A publication names an authored logical output at one exact build step.
// Output alias projection, physical producer selection, and final-bank
// selection are resolved from this coordinate rather than copied bindings.
struct PipelineBuildOutputCoordinate final {
  PipelinePublicationStepOrdinal step{};
  PipelineLogicalOutputOrdinal output{};
};

struct PipelineBuildOutputProjection final {
  PipelinePhysicalOutputOrdinal physical{};
  PipelineLogicalOutputOrdinal source{};
};

struct PipelineBuildWindowFinal final {
  PipelinePublicationStepOrdinal source_step{};
  std::uint32_t bank{};
};

struct PipelineBuildWindowAnchors final {
  PipelinePublicationStepOrdinal count_step{};
  PipelinePublicationStepOrdinal terminal_step{};
};

struct PipelineBuildPublicationBase final {
  PipelineBuildWindowControlOrdinal control{};
  PipelinePublicationStepOrdinal step{};
};

// One authored Window-control authority. Steps carry only this record's
// ordinal. Ordinary recurrence owns one anchor here; nested count/terminal
// anchors come only from PipelineBuildNestedWindow topology.
struct PipelineBuildWindowControl final {
  PipelinePublicationStepOrdinal ordinary_step{};
  std::size_t count_input{};
  std::size_t maximum{};
  std::size_t tile{};
  std::size_t terminal{NoWindowTerminal};
  std::uint32_t expected{1u};
  std::uint16_t nested{};
};

struct PipelineBuildStep final {
  std::shared_ptr<ProgramState> program;
  std::vector<PipelineBinding> inputs;
  std::vector<PipelineBinding> outputs;
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  PipelineBuildWindowControlOrdinal window_control{};
  std::uint16_t nested{};
  PipelineRoute route{PipelineRoute::Ordinary};
  // True only when repeat(...) retains this occurrence in caller-owned
  // iteration history. It prevents a terminal-only recurrence lowering from
  // eliding the authored intermediate writes.
  bool writes_each_iteration{};
};

struct PipelineBuildNestedWindow final {
  node::accel::detail::NestedTemplateShape shape{};
  // Exact leading Fold-output prefix that is also outer recurrent state.
  // Append-only window outputs follow this prefix and never enter a bank seal.
  std::size_t recurrent_output_count{};
};

struct PipelineBuildStatePair final {
  PipelineBinding published;
  PipelineBinding pending;
};

// Recurrent state and append-only window output both compute into private
// storage. Terminal publication is gated on complete Pipeline success. Window
// publication is count-gated after each successful Fold and is intentionally
// non-rollback: a later failure poisons a destination that may contain an
// already published prefix.
struct PipelineBuildPublicationEdge final {
  PipelineBinding target;
  PipelineBuildWindowControlOrdinal control{};
  PipelineLogicalOutputOrdinal output{};
};

struct PipelineBuildTerminalPublication final {
  PipelineBuildPublicationEdge edge;
};

struct PipelineBuildWindowPublication final {
  PipelineBuildPublicationEdge edge;
};

using PipelineBuildPublication = std::variant<PipelineBuildTerminalPublication,
                                              PipelineBuildWindowPublication>;

// Preparation keeps only execution metadata. Exact resource and View meaning
// lives in PipelineMemoryPlan::{resources,step_resources}; authored bindings
// never survive as a parallel post-plan authority.
struct PipelineFrozenStep final {
  std::shared_ptr<ProgramState> program;
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  std::uint16_t nested{};
  PipelineRoute route{PipelineRoute::Ordinary};
  bool writes_each_iteration{};
};

struct PipelineFrozenNestedWindow final {
  node::accel::detail::NestedTemplateShape shape{};
  std::size_t recurrent_output_count{};
};

struct PipelineBuildSnapshot final {
  std::shared_ptr<DeviceState> device;
  std::vector<PipelineFrozenStep> steps;
  std::vector<PipelineFrozenNestedWindow> nested_windows;
  std::size_t logical_step_count{};
  std::uint32_t sealed_repetitions{1u};
  PipelineProfile profile{PipelineProfile::None};
  bool commit{};
};

[[nodiscard]] inline const PipelineBuildPublicationEdge &
pipeline_publication_edge(const PipelineBuildPublication &publication) {
  return std::visit(
      [](const auto &typed) -> const PipelineBuildPublicationEdge & {
        return typed.edge;
      },
      publication);
}

[[nodiscard]] inline PipelineBuildPublicationEdge &
pipeline_publication_edge(PipelineBuildPublication &publication) {
  return std::visit(
      [](auto &typed) -> PipelineBuildPublicationEdge & { return typed.edge; },
      publication);
}

struct PipelineBuildState final {
  std::shared_ptr<DeviceState> device;
  // Keep the Pool/physical admission alive across every build-time Buffer
  // alias, including failure unwind.
  PipelineResidencyPlan residency{};
  std::vector<PipelineBuildStep> steps;
  std::vector<PipelineBuildStatePair> state_pairs;
  std::vector<PipelineBuildPublication> publications;
  std::vector<PipelineInternal> internals;
  std::vector<PipelineBuildWindowControl> window_controls;
  std::vector<PipelineBuildNestedWindow> nested_windows;
  // Cold-only physical owner table indexed by the frozen canonical resource
  // ordinal. Geometry and type remain owned by PipelineMemoryPlan.
  std::vector<std::shared_ptr<BufferState>> materialized_resources;
  std::shared_ptr<const PipelineMemoryPlan> memory;
  std::shared_ptr<StateSnapshotState> seed;
  std::shared_ptr<SnapshotStorageState> storage_seed;
  std::shared_ptr<PipelinePublicationState> device_seed;
  std::size_t binding_count{};
  std::size_t logical_step_count{};
  std::uint32_t sealed_repetitions{1u};
  PipelineProfile profile{PipelineProfile::None};
  std::uint64_t budget{};
  bool has_budget{};
  bool commit{};
  bool sealed{};
  bool sealed_repetitions_configured{};
  Reason failure{Reason::Ok};
};

// Sole build-state projection from a sealed Window-state ordinal to its
// constructor-closed nested topology. Consumers may copy projected values for
// a downstream handoff, but may not reconstruct K from the control scalars.
[[nodiscard]] inline const node::accel::detail::NestedTemplateShape *
pipeline_build_nested_shape(const PipelineBuildState &build,
                            const std::uint32_t state) noexcept {
  if (state >= build.window_controls.size()) {
    return nullptr;
  }
  const std::uint16_t nested = build.window_controls[state].nested;
  if (nested == 0u || nested > build.nested_windows.size()) {
    return nullptr;
  }
  return &build.nested_windows[nested - 1u].shape;
}

} // namespace rund::compute::detail
