#include "../internal.hpp"

#include "../endpoint_mode.hpp"

#include "../../../../../accel/context/local.hpp"
#include "../../../../../accel/context/transfer.hpp"
#include "../../../../type.hpp"

#include <limits>
#include <string_view>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

namespace accel = node::accel::detail;

[[nodiscard]] std::shared_ptr<BufferState>
resident_backing(const std::shared_ptr<VirtualBufferState> &buffer,
                 const std::shared_ptr<DeviceState> &device, const Type type,
                 const std::uint64_t bytes) noexcept {
  if (buffer == nullptr || buffer->backing == nullptr) {
    return {};
  }
  const std::shared_ptr<BufferState> &resident =
      VirtualBackingAccess::resident(*buffer->backing);
  if (resident == nullptr) {
    return {};
  }
  return endpoint_mode_detail::resident_matches(resident, device, type, bytes)
             ? resident
             : std::shared_ptr<BufferState>{};
}

[[nodiscard]] rund::AccelBuffer allocate_fallback(
    const AccelDeviceState &native, const std::uint64_t element_bytes,
    const std::uint64_t element_count, const rund::BufferUsage usage) noexcept {
  return accel::CreateAccelBufferWithInitialization(
      native.context,
      rund::AccelBufferDesc{.scalar_width_bytes = element_bytes,
                            .count = element_count,
                            .usage = usage},
      accel::BackendBufferInitialization::FullOverwrite,
      accel::BackendBufferMemory::HostVisiblePreferred);
}

[[nodiscard]] Status allocation_failure(const rund::AccelBuffer &buffer,
                                        const char *&reason) noexcept {
  const std::string_view native = buffer.check.reason == nullptr
                                      ? std::string_view{}
                                      : std::string_view{buffer.check.reason};
  const Reason mapped = project_reason(native, Reason::BackendUnsupported);
  if (mapped == Reason::DeviceCapacity || mapped == Reason::BufferCapacity ||
      mapped == Reason::PipelineMemoryBudget) {
    reason = "compute_pipeline_memory_budget";
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  if (mapped == Reason::BackendUnsupported || mapped == Reason::ReasonInvalid) {
    reason = "compute_backend_unsupported";
    return Status::fail(Reason::BackendUnsupported);
  }
  reason = buffer.check.reason == nullptr ? "compute_backend_unsupported"
                                          : buffer.check.reason;
  return Status::fail(mapped);
}

} // namespace

Status prepare_physical_buffers(VirtualPipelineState &state,
                                const VirtualRunProjection &run,
                                const AccelDeviceState &native,
                                DeviceVsmProductOwner &owner,
                                const VirtualDeviceVsmRouteProof &proof,
                                const char *&reason) noexcept {
  const std::size_t input_element_bytes = type_bytes(run.input_type);
  const std::size_t output_element_bytes = type_bytes(run.output_type);
  if (input_element_bytes == 0u || output_element_bytes == 0u ||
      run.active.input_bytes % input_element_bytes != 0u ||
      run.active.output_bytes % output_element_bytes != 0u ||
      run.active.input_bytes > std::numeric_limits<std::size_t>::max() ||
      run.active.output_bytes > std::numeric_limits<std::size_t>::max() ||
      run.input_count == 0u || run.input_count != state.input_count ||
      run.input_count > owner.inputs.size() || state.output == nullptr) {
    reason = "compute_pipeline_capacity";
    return Status::fail(Reason::PipelineCapacity);
  }
  if (!proof.valid()) {
    reason = "compute_backend_unsupported";
    return Status::fail(Reason::BackendUnsupported);
  }
  // Route admission already selected the immutable proof. The remaining
  // checks authenticate bytes, ranges, and mapped views.
  const bool graph_resident =
      proof.kind() == VirtualDeviceVsmRouteKind::GraphResident;
  const bool staged = proof.endpoint() == VirtualDeviceVsmEndpoint::Staged;
  const bool mapped = !graph_resident && staged;
  if (proof.kind() == VirtualDeviceVsmRouteKind::WindowRing &&
      proof.endpoint() == VirtualDeviceVsmEndpoint::Invalid) {
    reason = "compute_backend_unsupported";
    return Status::fail(Reason::BackendUnsupported);
  }
  EndpointSet endpoints{};
  const EndpointMode observed = graph_resident
                                    ? classify_endpoints(state, run, endpoints)
                                    : EndpointMode::Invalid;
  const EndpointMode sealed =
      proof.endpoint() == VirtualDeviceVsmEndpoint::Resident
          ? EndpointMode::Resident
          : EndpointMode::Staged;
  if (graph_resident &&
      (observed == EndpointMode::Invalid || observed != sealed)) {
    reason = "device_vsm_graph_resident_external_binding_invalid";
    return Status::fail(Reason::BackendUnsupported);
  }
  owner.input_count = run.input_count;
  owner.route_proof = proof;
  const std::uint64_t input_count =
      run.active.input_bytes / input_element_bytes;
  for (std::size_t index = 0u; index < owner.input_count; ++index) {
    owner.resident_inputs[index] =
        graph_resident
            ? (proof.endpoint() == VirtualDeviceVsmEndpoint::Resident
                   ? endpoints.inputs[index]
                   : std::shared_ptr<BufferState>{})
            : (run.clip_window
                   ? std::shared_ptr<BufferState>{}
                   : resident_backing(state.inputs[index],
                                      state.pipeline->device, run.input_type,
                                      run.active.input_bytes));
    if (proof.kind() == VirtualDeviceVsmRouteKind::WindowRing &&
        (owner.resident_inputs[index] != nullptr) !=
            (proof.endpoint() == VirtualDeviceVsmEndpoint::Resident)) {
      reason = "compute_backend_unsupported";
      return Status::fail(Reason::BackendUnsupported);
    }
    if (owner.resident_inputs[index] != nullptr) {
      const auto &source = accel_buffer(*owner.resident_inputs[index])->buffer;
      owner.inputs[index] = accel::ProjectAccelBufferView(
          native.context, source,
          rund::AccelBufferDesc{.scalar_width_bytes = source.scalar_width_bytes,
                                .count = source.count,
                                .usage = rund::BufferUsage::ReadOnly});
      if (!owner.inputs[index]) {
        reason = "compute_backend_unsupported";
        return Status::fail(Reason::BackendUnsupported);
      }
      continue;
    }
    owner.inputs[index] =
        allocate_fallback(native, input_element_bytes, input_count,
                          run.clip_window ? rund::BufferUsage::ReadWrite
                                          : rund::BufferUsage::ReadOnly);
    if (!owner.inputs[index]) {
      return allocation_failure(owner.inputs[index], reason);
    }
    if (mapped) {
      const accel::AccelHostWriteView view =
          accel::WriteAccelBuffer(native.context, owner.inputs[index]);
      if (!view || view.bytes < run.active.input_bytes) {
        reason = "device_vsm_staged_host_mapping_unavailable";
        return Status::fail(Reason::BackendUnsupported);
      }
    } else {
      owner.input_staging[index].resize(
          static_cast<std::size_t>(run.active.input_bytes));
    }
  }

  owner.resident_output =
      graph_resident
          ? (proof.endpoint() == VirtualDeviceVsmEndpoint::Resident
                 ? endpoints.output
                 : std::shared_ptr<BufferState>{})
          : resident_backing(state.output, state.pipeline->device,
                             run.output_type, run.active.output_bytes);
  if (proof.kind() == VirtualDeviceVsmRouteKind::WindowRing &&
      (owner.resident_output != nullptr) !=
          (proof.endpoint() == VirtualDeviceVsmEndpoint::Resident)) {
    reason = "compute_backend_unsupported";
    return Status::fail(Reason::BackendUnsupported);
  }
  if (owner.resident_output != nullptr) {
    const auto &source = accel_buffer(*owner.resident_output)->buffer;
    owner.output = accel::ProjectAccelBufferView(
        native.context, source,
        rund::AccelBufferDesc{.scalar_width_bytes = source.scalar_width_bytes,
                              .count = source.count,
                              .usage = rund::BufferUsage::WriteOnly});
    // Exact output-hash publication requires a stable Host observation view;
    // otherwise the established fallback retains one whole-run readback.
    if (!owner.output ||
        !accel::ReadAccelBuffer(native.context, owner.output)) {
      if (graph_resident &&
          proof.endpoint() == VirtualDeviceVsmEndpoint::Resident) {
        reason = "device_vsm_graph_resident_external_binding_invalid";
        return Status::fail(Reason::BackendUnsupported);
      }
      owner.resident_output.reset();
      owner.output = {};
    }
  }
  if (owner.resident_output == nullptr) {
    owner.output =
        allocate_fallback(native, output_element_bytes,
                          run.active.output_bytes / output_element_bytes,
                          rund::BufferUsage::WriteOnly);
    if (!owner.output) {
      return allocation_failure(owner.output, reason);
    }
    if (mapped) {
      const accel::AccelHostView view =
          accel::ReadAccelBuffer(native.context, owner.output);
      if (!view || view.bytes < run.active.output_bytes) {
        reason = "device_vsm_staged_host_mapping_unavailable";
        return Status::fail(Reason::BackendUnsupported);
      }
    } else {
      owner.output_staging.resize(
          static_cast<std::size_t>(run.active.output_bytes));
    }
  }
  reason = "ok";
  return Status::success();
}

} // namespace rund::compute::detail::device_vsm_product_detail
