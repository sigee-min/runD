#include "internal.hpp"

#include "../../../recurrence/match.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::device_vsm_projection {

rund::AccelCheck ValidateProjectionCandidate(
    const DeviceVsmProofRequest &request, const Candidate &candidate) noexcept {
  if (candidate.pointwise) {
    const std::uint64_t input_buffers =
        candidate.first->planned->plan.input_buffer_count;
    std::uint64_t input_bytes_per_tile = 0u;
    if (!::rund::kernel::checked::mul(input_buffers,
                                     request.geometry.element_bytes,
                                     input_bytes_per_tile) ||
        candidate.first->planned->plan.tile_count !=
            request.geometry.payload_bytes / request.geometry.element_bytes ||
        candidate.first->planned->plan.input_bytes_per_tile !=
            input_bytes_per_tile ||
        candidate.first->planned->plan.output_bytes_per_tile !=
            request.geometry.element_bytes ||
        request.residents.input_count != input_buffers ||
        request.residents.output_count != 1u) {
      return {false, "device_vsm_pipeline_geometry_invalid"};
    }
    if (request.peer_pipeline.ok &&
        (candidate.peer == nullptr ||
         !same_pointwise_pipeline(*candidate.peer, *candidate.first,
                                  candidate.bindings))) {
      return {false, "device_vsm_peer_pipeline_invalid"};
    }
  } else if (candidate.graph_pointwise) {
    const rund::kernel::ComputePlan &first_plan =
        candidate.graph_pointwise_steps[0u]->planned->plan;
    bool chain = request.residents.input_count ==
                     request.graph_pointwise_topology.external_input_count &&
                 request.residents.output_count == 1u;
    std::array<std::size_t, DeviceVsmResidentCapacity> external_stages{};
    std::array<std::size_t, DeviceVsmResidentCapacity> external_inputs{};
    external_stages.fill(DeviceVsmGraphStageCapacity);
    external_inputs.fill(DeviceVsmGraphStageInputCapacity);
    for (std::size_t stage = 0u;
         chain && stage < request.graph_pointwise_pipeline_count; ++stage) {
      const rund::kernel::ComputePlan &candidate_plan =
          candidate.graph_pointwise_steps[stage]->planned->plan;
      const DeviceVsmGraphPointwiseStageTopology &topology =
          request.graph_pointwise_topology.stages[stage];
      std::uint64_t input_bytes_per_tile = 0u;
      chain =
          candidate_plan.input_buffer_count == topology.input_count &&
          candidate_plan.output_buffer_count == 1u &&
          candidate_plan.api == first_plan.api &&
          candidate_plan.scalar == first_plan.scalar &&
          candidate_plan.domain == first_plan.domain &&
          candidate_plan.tile_count ==
              request.geometry.payload_bytes / request.geometry.element_bytes &&
          ::rund::kernel::checked::mul(topology.input_count,
                                       request.geometry.element_bytes,
                                       input_bytes_per_tile) &&
          candidate_plan.input_bytes_per_tile == input_bytes_per_tile &&
          candidate_plan.output_bytes_per_tile ==
              request.geometry.element_bytes &&
          candidate_plan.param_bytes == 0u;
      for (std::size_t input = 0u; chain && input < topology.input_count;
           ++input) {
        const DeviceVsmGraphValueSource source = topology.inputs[input];
        if (source.kind == DeviceVsmGraphValueSourceKind::ExternalInput) {
          if (external_stages[source.index] == DeviceVsmGraphStageCapacity) {
            external_stages[source.index] = stage;
            external_inputs[source.index] = input;
          } else {
            chain = SameView(
                candidate.graph_pointwise_bindings[external_stages[source.index]]
                    .resident_inputs,
                external_inputs[source.index],
                candidate.graph_pointwise_bindings[stage].resident_inputs,
                input);
          }
        } else {
          chain =
              source.kind == DeviceVsmGraphValueSourceKind::StageOutput &&
              SameView(
                  candidate.graph_pointwise_bindings[source.index]
                      .resident_outputs,
                  0u, candidate.graph_pointwise_bindings[stage].resident_inputs,
                  input);
        }
      }
    }
    if (!chain) {
      return {false, "device_vsm_graph_pointwise_chain_invalid"};
    }
  } else if (candidate.graph_collective_reduce) {
    const std::uint64_t input_buffers =
        candidate.graph_map->planned->plan.input_buffer_count;
    std::uint64_t input_bytes_per_tile = 0u;
    if (!::rund::kernel::checked::mul(input_buffers,
                                     request.geometry.element_bytes,
                                     input_bytes_per_tile) ||
        request.output_bytes != sizeof(std::uint64_t) || input_buffers == 0u ||
        input_buffers >= DeviceVsmResidentCapacity ||
        request.residents.input_count != input_buffers ||
        request.residents.output_count != 1u ||
        candidate.graph_map->planned->plan.tile_count !=
            request.geometry.payload_bytes / request.geometry.element_bytes ||
        candidate.graph_map->planned->plan.input_bytes_per_tile !=
            input_bytes_per_tile ||
        candidate.graph_map->planned->plan.output_bytes_per_tile !=
            request.geometry.element_bytes ||
        candidate.graph_reduce.api != candidate.graph_map->planned->plan.api ||
        candidate.graph_reduce.semantic.element !=
            rund::kernel::ReduceElement::U64 ||
        candidate.graph_reduce.semantic.element_count !=
            request.geometry.frame_bytes / request.geometry.element_bytes ||
        (request.peer_pipeline.ok &&
         (candidate.peer == nullptr ||
          !same_graph_map_pipeline(*candidate.peer, *candidate.graph_map,
                                   candidate.graph_bindings)))) {
      return {false, "device_vsm_graph_peer_invalid"};
    }
  } else if (candidate.scanned) {
    const std::uint64_t input_buffers =
        candidate.scan.map.kind == DeviceVsmScanMapKind::CanonicalTotalU64 &&
                candidate.scan.map_step != nullptr &&
                candidate.scan.map_step->planned != nullptr
            ? candidate.scan.map_step->planned->plan.input_buffer_count
            : 1u;
    if (request.residents.input_count != input_buffers ||
        request.residents.output_count != 1u ||
        request.output_bytes != request.geometry.logical_bytes ||
        candidate.scan.semantic.element_count !=
            request.geometry.frame_bytes / request.geometry.element_bytes ||
        (request.peer_pipeline.ok &&
         (candidate.peer == nullptr ||
          !device_vsm_scan_projection::same(*candidate.peer,
                                            candidate.scan)))) {
      return {false, "device_vsm_scan_peer_invalid"};
    }
  } else if (candidate.reduced) {
    if (request.output_bytes != request.geometry.element_bytes ||
        candidate.reduce.semantic.element_count !=
            request.geometry.frame_bytes / request.geometry.element_bytes ||
        (request.peer_pipeline.ok &&
         (candidate.peer == nullptr ||
          !device_vsm_reduce_projection::same(*candidate.peer,
                                              candidate.reduce)))) {
      return {false, "device_vsm_reduce_peer_invalid"};
    }
  } else if (request.peer_pipeline.ok &&
             (candidate.peer == nullptr ||
              !device_vsm_window_projection::same_window_pipeline(
                  *candidate.peer, candidate.window))) {
    return {false, "device_vsm_window_peer_invalid"};
  }
  return {true, "ok"};
}

} // namespace rund::node::accel::detail::device_vsm_projection
