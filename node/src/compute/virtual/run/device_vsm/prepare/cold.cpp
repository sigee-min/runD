#include "../../../../../accel/kernel/prepared/pipeline.hpp"
#include "../../../../../accel/kernel/prepared/run.hpp"
#include "../internal.hpp"

#include "graph_pointwise_topology.hpp"
#include "graph_wavefront.hpp"

#include "../../../../../accel/context/local.hpp"
#include "../../../../../accel/context/transfer.hpp"
#include "../../../../type.hpp"

#include <new>
#include <string_view>

namespace rund::compute::detail::device_vsm_product_detail {

namespace {

[[nodiscard]] residency::RegistrationResult
release_registration(residency::DeviceVsmRegistration *const registration) noexcept {
  return registration == nullptr ? residency::RegistrationResult::Invalid
                                 : registration->release();
}

} // namespace

std::shared_ptr<DeviceVsmProductOwner>
prepare_cold_owner(VirtualPipelineState &state, const VirtualRunProjection &run,
                   const AccelDeviceState &native,
                   const node::accel::detail::DeviceVsmIdentity &identity,
                   const VirtualDeviceVsmRouteProof &proof,
                   Status &status, const char *&reason) noexcept {
  namespace accel = node::accel::detail;
  try {
    if (!proof.valid()) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "compute_backend_unsupported";
      return {};
    }
    const bool graph_resident =
        proof.kind == VirtualDeviceVsmRouteKind::GraphResident;
    const bool window_ring =
        proof.kind == VirtualDeviceVsmRouteKind::WindowRing;
    const bool graph_pointwise =
        run.graph_execution && !run.graph_reduction &&
        state.geometry.route == VirtualRoute::GraphPointwise && !graph_resident;
    storage::Reservation capacity{};
    status = reserve_device_vsm_capacity(state, capacity);
    if (!status) {
      reason = "compute_device_pipeline_memory_capacity";
      return {};
    }
    auto owner = std::make_shared<DeviceVsmProductOwner>();
    owner->capacity = std::move(capacity);
    owner->route_proof = proof;
    if (!select_pipelines(state, run, owner->pipelines, owner->pipeline_stages,
                          owner->pipeline_count)) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "compute_backend_unsupported";
      return {};
    }
    status =
        prepare_physical_buffers(state, run, native, *owner, proof, reason);
    if (!status) {
      return {};
    }
    if (graph_resident &&
        !graph_resident_bindings_match(state, run, native, *owner, nullptr)) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "device_vsm_graph_resident_external_binding_invalid";
      return {};
    }
    const std::size_t element_bytes = type_bytes(run.input_type);
    std::array<accel::UploadRoute, VirtualPipelineState::InputCapacity>
        input_routes{};
    accel::DeviceVsmResidentSet residents{};
    for (std::size_t index = 0u; index < owner->input_count; ++index) {
      input_routes[index] =
          accel::ProjectAccelBufferRoute(native.context, owner->inputs[index]);
      if (input_routes[index].handle == nullptr) {
        status = Status::fail(Reason::BackendUnsupported);
        reason = "compute_backend_unsupported";
        return {};
      }
      residents.rows[index] = accel::DeviceVsmResidentBinding{
          .role = accel::DeviceVsmResidentRole::Input,
          .backing = input_routes[index].resident,
          .handle = input_routes[index].handle,
      };
    }
    const accel::UploadRoute output =
        accel::ProjectAccelBufferRoute(native.context, owner->output);
    if (output.handle == nullptr) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "compute_backend_unsupported";
      return {};
    }
    residents.rows[owner->input_count] = accel::DeviceVsmResidentBinding{
        .role = accel::DeviceVsmResidentRole::Output,
        .backing = output.resident,
        .handle = output.handle,
    };
    residents.count = static_cast<std::uint32_t>(owner->input_count + 1u);
    residents.input_count = static_cast<std::uint32_t>(owner->input_count);
    residents.output_count = 1u;
    const accel::DeviceVsmPageGeometry geometry{
        .logical_bytes = run.active.input_bytes,
        .payload_bytes = run.input_payload_bytes,
        .frame_bytes = run.input_page_bytes,
        .read_prefix_bytes = run.input_prefix_bytes,
        .target_offset_bytes = run.input_prefix_bytes,
        .read_suffix_bytes = run.input_page_bytes - run.input_prefix_bytes -
                             run.input_payload_bytes,
        .page_count = run.graph_execution ? run.active.graph.page_count()
                                          : run.active.stream.page_count(),
        .element_bytes = static_cast<std::uint32_t>(element_bytes),
    };
    accel::DeviceVsmGraphWavefrontProof graph_wavefront{};
    if (run.graph_execution &&
        !project_graph_wavefront(
            state.pipeline->residency->tiled_graph(), geometry.page_count,
            owner->pipeline_stages[0u],
            owner->pipeline_stages[owner->pipeline_count - 1u],
            graph_wavefront)) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "device_vsm_graph_wavefront_invalid";
      return {};
    }
    accel::DeviceVsmGraphPointwiseTopology graph_pointwise_topology{};
    if (graph_pointwise && !project_graph_pointwise_topology(
                               state.pipeline->residency->tiled_graph(),
                               {state.graph_input_resources.data(),
                                state.graph_input_resource_count},
                               graph_pointwise_topology)) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "device_vsm_graph_pointwise_topology_invalid";
      return {};
    }
    accel::DeviceVsmPageMap graph_pointwise_page_map{};
    if (graph_pointwise &&
        !project_graph_page_map(state.pipeline->residency->tiled_graph(),
                                {state.graph_input_resources.data(),
                                 state.graph_input_resource_count},
                                graph_pointwise_page_map)) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = "device_vsm_graph_pointwise_page_map_invalid";
      return {};
    }
    accel::DeviceVsmGraphResidentProof graph_resident_proof{};
    const char *graph_resident_reason = nullptr;
    if (graph_resident &&
        (state.pipeline->residency_pool == nullptr ||
         !project_graph_resident(state.pipeline->residency->tiled_graph(),
                                 *state.pipeline->residency_pool,
                                 {state.graph_input_resources.data(),
                                  state.graph_input_resource_count},
                                 graph_wavefront, graph_resident_proof,
                                 graph_resident_reason))) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = graph_resident_reason == nullptr
                   ? "device_vsm_graph_resident_projection_invalid"
                   : graph_resident_reason;
      return {};
    }
    const std::shared_ptr<PipelineState> terminal_pipeline =
        graph_terminal_pipeline(state, 0u);
    accel::DeviceVsmProofRequest proof_request{
        .pipeline = owner->pipelines[0u]->prepared,
        .peer_pipeline = graph_pointwise ? owner->pipelines[1u]->prepared
                         : (graph_resident || run.graph_reduction)
                             ? accel::PreparedKernelPipeline{}
                             : state.alternate_pipeline->prepared,
        .graph_collective_pipeline = run.graph_reduction
                                         ? terminal_pipeline->prepared
                                         : accel::PreparedKernelPipeline{},
        .graph_pointwise_page_map = graph_pointwise_page_map,
        .graph_resident = graph_resident_proof,
        .identity = identity,
        .residents = std::move(residents),
        .geometry = geometry,
        .output_bytes = run.active.output_bytes,
        .graph_wavefront = graph_wavefront,
        .width = 2u,
        .kind = graph_resident ? accel::DeviceVsmProjectionKind::GraphResident
                : window_ring  ? accel::DeviceVsmProjectionKind::WindowRing
                : graph_pointwise
                    ? accel::DeviceVsmProjectionKind::GraphPointwise
                : run.graph_reduction
                    ? accel::DeviceVsmProjectionKind::GraphMapReduce
                : run.scan      ? accel::DeviceVsmProjectionKind::Scan
                : run.reduction ? accel::DeviceVsmProjectionKind::Reduce
                                : accel::DeviceVsmProjectionKind::Direct,
    };
    if (graph_pointwise) {
      proof_request.graph_pointwise_pipeline_count =
          static_cast<std::uint8_t>(owner->pipeline_count);
      proof_request.graph_pointwise_topology = graph_pointwise_topology;
      for (std::size_t stage = 0u; stage < owner->pipeline_count; ++stage) {
        proof_request.graph_pointwise_pipelines[stage] =
            owner->pipelines[stage]->prepared;
      }
    }
    if (graph_resident) {
      proof_request.graph_resident_pipeline_count =
          static_cast<std::uint8_t>(owner->pipeline_count);
      for (std::size_t stage = 0u; stage < owner->pipeline_count; ++stage) {
        proof_request.graph_resident_pipelines[stage] =
            owner->pipelines[stage]->prepared;
      }
    }
    const accel::DeviceVsmProofProjection projected =
        accel::ProjectPreparedKernelPipelineDeviceVsm(proof_request);
    if (!projected) {
      status = Status::fail(Reason::BackendUnsupported);
      reason = projected.check.reason;
      return {};
    }
    owner->proof = projected.proof;
    owner->preparation = accel::PrepareDeviceVsm(native.pick, owner->proof);
    if (!owner->preparation) {
      reason = owner->preparation.capability.check.reason;
      status = Status::fail(project_reason(
          reason == nullptr ? std::string_view{} : std::string_view{reason},
          Reason::BackendUnsupported));
      return {};
    }
    status = commit_device_vsm_memory(*owner);
    if (!status) {
      reason = status.reason() == Reason::DevicePipelineMemoryCapacity
                   ? "compute_device_pipeline_memory_capacity"
                   : "compute_pipeline_memory_budget";
      return {};
    }
    owner->registration = residency::register_device_vsm_proof(
        state.pipeline->device->residency, owner->proof);
    if (owner->registration == nullptr) {
      status = Status::fail(Reason::PipelineMemoryBudget);
      reason = "compute_pipeline_capacity";
      return {};
    }
    owner->release_registration = &release_registration;
    owner->evidence = std::make_shared<DeviceVsmProductEvidence>();
    classify_device_vsm_backing_evidence(*owner, *owner->evidence);
    owner->cold_prepare_count = 1u;
    status = Status::success();
    reason = "ok";
    return owner;
  } catch (const std::bad_alloc &) {
    status = Status::fail(Reason::PipelineMemoryBudget);
    reason = "compute_pipeline_memory_budget";
    return {};
  } catch (const std::length_error &) {
    status = Status::fail(Reason::PipelineMemoryBudget);
    reason = "compute_pipeline_memory_budget";
    return {};
  }
}

} // namespace rund::compute::detail::device_vsm_product_detail
