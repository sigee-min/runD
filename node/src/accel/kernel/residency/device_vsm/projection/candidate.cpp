#include "internal.hpp"

#include "graph_resident.hpp"

#include "../validation.hpp"

namespace rund::node::accel::detail::device_vsm_projection {

bool SelectProjectionCandidate(const DeviceVsmProofRequest &request,
                               Candidate &candidate) noexcept {
  candidate = {};
  candidate.pipeline = static_cast<const prepared::PipelineState *>(
      request.pipeline.owner.get());
  if (!request.pipeline.ok || candidate.pipeline == nullptr ||
      !request.identity || request.residents.count == 0u ||
      !device_vsm_runtime_geometry_valid(request.geometry) || request.width < 2u ||
      request.width > 4u || request.width > request.geometry.page_count) {
    candidate.check = {false, "device_vsm_projection_request_invalid"};
    return false;
  }

  candidate.primary_reason = nullptr;
  candidate.pointwise =
      request.kind == DeviceVsmProjectionKind::Direct &&
      exact_pointwise_pipeline(*candidate.pipeline, candidate.first,
                               candidate.bindings, candidate.primary_reason) &&
      candidate.first != nullptr;
  candidate.graph_reason = nullptr;
  candidate.graph_map = nullptr;
  candidate.graph_map_reduce =
      request.kind == DeviceVsmProjectionKind::GraphMapReduce &&
      device_vsm_graph_wavefront_valid(request.graph_wavefront,
                                       request.geometry.page_count) &&
      exact_graph_map_pipeline(*candidate.pipeline, candidate.graph_map,
                               candidate.graph_bindings,
                               candidate.graph_reason) &&
      candidate.graph_map != nullptr;

  candidate.peer =
      request.peer_pipeline.ok
          ? static_cast<const prepared::PipelineState *>(
                request.peer_pipeline.owner.get())
          : nullptr;
  candidate.graph_resident_proof = request.graph_resident;
  const auto graph_pick = device_vsm_graph_projection::pick_graph(
      request, exact_graph_map_pipeline);
  candidate.graph_resident_ready = graph_pick.ready;
  candidate.graph_resident_reason = graph_pick.reason;
  candidate.graph_resident_proof = graph_pick.proof;
  candidate.graph_resident_pipelines = graph_pick.pipelines;
  candidate.graph_resident_states = graph_pick.states;
  candidate.graph_resident_steps = graph_pick.steps;
  candidate.graph_resident_bindings = graph_pick.bindings;
  if (request.kind == DeviceVsmProjectionKind::GraphResident &&
      !candidate.graph_resident_ready) {
    candidate.check = {false, candidate.graph_resident_reason};
    return false;
  }

  candidate.graph_pointwise_reason =
      "device_vsm_graph_pointwise_stage_invalid";
  candidate.graph_pointwise =
      request.kind == DeviceVsmProjectionKind::GraphPointwise &&
      request.graph_pointwise_pipeline_count >= 2u &&
      request.graph_pointwise_pipeline_count <= DeviceVsmGraphStageCapacity &&
      request.graph_pointwise_pipeline_count ==
          request.graph_wavefront.stage_count &&
      request.graph_pointwise_pipeline_count ==
          request.graph_pointwise_topology.stage_count &&
      device_vsm_graph_pointwise_topology_valid(
          request.graph_pointwise_topology, request.graph_wavefront) &&
      device_vsm_graph_wavefront_valid(request.graph_wavefront,
                                       request.geometry.page_count);
  for (std::size_t stage = 0u;
       candidate.graph_pointwise &&
       stage < request.graph_pointwise_pipeline_count;
       ++stage) {
    const PreparedKernelPipeline &prepared_pipeline =
        request.graph_pointwise_pipelines[stage];
    candidate.graph_pointwise_pipelines[stage] =
        prepared_pipeline.ok
            ? static_cast<const prepared::PipelineState *>(
                  prepared_pipeline.owner.get())
            : nullptr;
    candidate.graph_pointwise =
        candidate.graph_pointwise_pipelines[stage] != nullptr &&
        exact_graph_map_pipeline(
            *candidate.graph_pointwise_pipelines[stage],
            candidate.graph_pointwise_steps[stage],
            candidate.graph_pointwise_bindings[stage],
            candidate.graph_pointwise_reason) &&
        candidate.graph_pointwise_steps[stage] != nullptr;
  }

  candidate.graph_collective =
      request.graph_collective_pipeline.ok
          ? static_cast<const prepared::PipelineState *>(
                request.graph_collective_pipeline.owner.get())
          : nullptr;
  candidate.graph_reduce_reason = nullptr;
  candidate.graph_collective_reduce =
      candidate.graph_map_reduce && candidate.graph_collective != nullptr &&
      device_vsm_reduce_projection::exact_graph(
          *candidate.graph_collective, candidate.graph_reduce,
          candidate.graph_reduce_reason);

  candidate.scan_reason = nullptr;
  candidate.scanned = request.kind == DeviceVsmProjectionKind::Scan &&
                      device_vsm_scan_projection::exact(
                          *candidate.pipeline, candidate.scan,
                          candidate.scan_reason);
  candidate.reduce_reason = nullptr;
  candidate.reduced = request.kind == DeviceVsmProjectionKind::Reduce &&
                      device_vsm_reduce_projection::exact(
                          *candidate.pipeline, candidate.reduce,
                          candidate.reduce_reason);

  candidate.window_reason = nullptr;
  if (request.kind == DeviceVsmProjectionKind::WindowRing &&
      !device_vsm_window_ring_plan_expected(request.geometry,
                                             candidate.ring_plan)) {
    candidate.check = {false, "device_vsm_window_ring_plan_invalid"};
    return false;
  }
  candidate.windowed =
      (request.kind == DeviceVsmProjectionKind::Direct ||
       request.kind == DeviceVsmProjectionKind::WindowRing) &&
      !candidate.pointwise &&
      device_vsm_window_projection::exact_window_pipeline(
          *candidate.pipeline, candidate.window, candidate.window_reason);
  if (!candidate.pointwise && !candidate.graph_resident_ready &&
      !candidate.graph_pointwise && !candidate.graph_collective_reduce &&
      !candidate.scanned && !candidate.reduced && !candidate.windowed) {
    const char *const reason =
        request.kind == DeviceVsmProjectionKind::GraphPointwise
            ? candidate.graph_pointwise_reason
        : request.kind == DeviceVsmProjectionKind::GraphResident
            ? candidate.graph_resident_reason
        : request.kind == DeviceVsmProjectionKind::GraphMapReduce
            ? (candidate.graph_map_reduce ? candidate.graph_reduce_reason
                                          : candidate.graph_reason)
        : request.kind == DeviceVsmProjectionKind::Scan ? candidate.scan_reason
        : request.kind == DeviceVsmProjectionKind::Reduce
            ? candidate.reduce_reason
        : candidate.window_reason == nullptr ? candidate.primary_reason
                                             : candidate.window_reason;
    candidate.check = {false, reason};
    return false;
  }
  candidate.check = {true, "ok"};
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_projection
