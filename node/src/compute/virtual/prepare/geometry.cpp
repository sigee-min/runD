#include "internal.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/reduce/operation.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/window/model.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace rund::compute::detail::virtual_prepare_detail {

std::optional<VirtualGeometry>
classify_geometry(const ProgramState &program) noexcept {
  if (program.graph_info.nodes.empty() || program.input_sizes.empty() ||
      program.input_sizes.size() > VirtualPipelineState::InputCapacity ||
      program.output_sizes.size() != 1u || program.input_sizes[0] == 0u ||
      program.output_sizes[0] == 0u ||
      !std::all_of(program.input_sizes.begin(), program.input_sizes.end(),
                   [&program](const std::size_t count) {
                     return count == program.input_sizes.front();
                   })) {
    return std::nullopt;
  }
  const std::uint64_t input_elements = program.input_sizes[0];
  const std::uint64_t output_elements = program.output_sizes[0];
  const graph::Node *collective = nullptr;
  bool map_before_collective = false;
  bool map_after_collective = false;
  for (const graph::Node &node : program.graph_info.nodes) {
    if (node.operation == graph::Operation::Map) {
      // A pointwise transform after a collective changes the meaning of each
      // page partial and cannot be moved across the global merge.
      if (collective != nullptr &&
          collective->operation == graph::Operation::Reduce) {
        return std::nullopt;
      }
      map_before_collective = map_before_collective || collective == nullptr;
      map_after_collective = map_after_collective || collective != nullptr;
      continue;
    }
    if ((node.operation != graph::Operation::Window &&
         node.operation != graph::Operation::Reduce &&
         node.operation != graph::Operation::Scan) ||
        collective != nullptr) {
      return std::nullopt;
    }
    collective = &node;
  }
  if (collective == nullptr) {
    return input_elements == output_elements
               ? std::optional<VirtualGeometry>{VirtualGeometry{
                     .route = VirtualRoute::Pointwise,
                     .input_payload_elements = input_elements,
                     .output_payload_elements = output_elements,
                     .input_frame_elements = input_elements,
                     .output_frame_elements = output_elements,
                 }}
               : std::nullopt;
  }
  const graph::Footprint &footprint = collective->footprint;
  if (collective->operation == graph::Operation::Scan) {
    const auto operation = static_cast<kernel::ScanOp>(footprint.operation);
    const bool exclusive = operation == kernel::ScanOp::ExclusiveSum;
    const bool inclusive = operation == kernel::ScanOp::InclusiveSum;
    // The graph walk above already proves that every non-Scan node is a Map
    // and that no Map follows the sole collective. Requiring exactly one Map
    // here incorrectly rejects a canonical pure fanout/fan-in Map DAG that
    // common lowering can retain as one typed semantic owner for DeviceVsm.
    if (map_after_collective ||
        footprint.pattern != graph::AccessPattern::Prefix ||
        footprint.input_elements != input_elements ||
        footprint.output_elements != output_elements ||
        input_elements != output_elements || footprint.tile_elements == 0u ||
        (!exclusive && !inclusive) || (exclusive && input_elements < 2u)) {
      return std::nullopt;
    }
    return VirtualGeometry{
        .route = VirtualRoute::Scan,
        .input_payload_elements =
            exclusive ? input_elements - 1u : input_elements,
        .output_payload_elements =
            exclusive ? output_elements - 1u : output_elements,
        .input_frame_elements = input_elements,
        .output_frame_elements = output_elements,
        .input_prefix_elements = exclusive ? 1u : 0u,
        .output_prefix_elements = exclusive ? 1u : 0u,
        .operation = footprint.operation,
        .device_vsm_required =
            map_before_collective || program.input_sizes.size() > 1u,
        .materialization_hi = program.graph_info.fingerprint.hi,
        .materialization_lo = program.graph_info.fingerprint.lo,
    };
  }
  if (collective->operation == graph::Operation::Reduce) {
    const auto operation = static_cast<kernel::ReduceOp>(footprint.operation);
    const bool unsigned_sum = operation != kernel::ReduceOp::Sum ||
                              program.input_types[0] == Type::U32 ||
                              program.input_types[0] == Type::U64;
    // Boundary padding is injected before the compiled graph executes. An
    // arbitrary preceding Map can transform the Reduce identity (for example
    // f(0)=1), so accepting composition would count inactive tail elements.
    // A later Map would be applied to per-page partials instead of once to the
    // global result. Both require a split global execution stage.
    if ((program.input_sizes.size() != 1u && !map_before_collective) ||
        (!map_before_collective && program.graph_info.nodes.size() != 1u) ||
        footprint.pattern != graph::AccessPattern::Reduction ||
        footprint.input_elements != input_elements ||
        footprint.output_elements != 1u || output_elements != 1u ||
        footprint.tile_elements == 0u || !kernel::reduce::valid(operation) ||
        !unsigned_sum) {
      return std::nullopt;
    }
    return VirtualGeometry{
        .route = map_before_collective ? VirtualRoute::GraphReduction
                                       : VirtualRoute::Reduction,
        .input_payload_elements = input_elements,
        .output_payload_elements = 1u,
        .input_frame_elements = input_elements,
        .output_frame_elements = 1u,
        .intermediate_frame_elements =
            map_before_collective ? input_elements : 0u,
        .operation = footprint.operation,
        .device_vsm_required = program.input_sizes.size() != 1u,
        .materialization_hi = program.graph_info.fingerprint.hi,
        .materialization_lo = program.graph_info.fingerprint.lo,
    };
  }
  std::uint64_t symmetric_window = 0u;
  if (program.input_sizes.size() != 1u ||
      footprint.pattern != graph::AccessPattern::Window ||
      footprint.input_elements != input_elements ||
      footprint.output_elements != output_elements ||
      input_elements != output_elements || footprint.stride != 1u ||
      footprint.pad_left == 0u ||
      !kernel::checked::mul(footprint.pad_left, 2u, symmetric_window) ||
      !kernel::checked::add(symmetric_window, 1u, symmetric_window) ||
      footprint.window_size != symmetric_window ||
      input_elements <= footprint.pad_left * 2u ||
      (footprint.boundary !=
           static_cast<std::uint32_t>(kernel::WindowBoundary::Clamp) &&
       footprint.boundary !=
           static_cast<std::uint32_t>(kernel::WindowBoundary::Clip))) {
    return std::nullopt;
  }
  // The rolling Clip path pads the physical frame with the Window identity.
  // A preceding Map may change that identity. Preserve the public semantic
  // shape, but seal it as DeviceVsm-only so an unsupported exact GPU proof can
  // never fall through to the padding-based Host-service implementation.
  const bool device_vsm_required =
      footprint.boundary ==
          static_cast<std::uint32_t>(kernel::WindowBoundary::Clip) &&
      map_before_collective;
  return VirtualGeometry{
      .route = VirtualRoute::Window,
      .input_payload_elements = input_elements - footprint.pad_left * 2u,
      .output_payload_elements = output_elements - footprint.pad_left * 2u,
      .input_frame_elements = input_elements,
      .output_frame_elements = output_elements,
      .input_prefix_elements = footprint.pad_left,
      .output_prefix_elements = footprint.pad_left,
      .operation = footprint.operation,
      .boundary = footprint.boundary,
      .device_vsm_required = device_vsm_required,
      .materialization_hi = program.graph_info.fingerprint.hi,
      .materialization_lo = program.graph_info.fingerprint.lo,
  };
}

} // namespace rund::compute::detail::virtual_prepare_detail
