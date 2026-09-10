#include "../../../../kernel/backend/execute.hpp"
#include "../../../../kernel/backend/source/storage.hpp"
#include "../../../../kernel/backend/template/arithmetic.hpp"
#include "../../../../kernel/recurrence/plan.hpp"
#include "../../../../kernel/status.hpp"
#include "../../../../resident/window/admission/runtime/windows.hpp"

#include "../../../collective/chunk.hpp"
#include "../../../compact/local.hpp"
#include "../../../descriptor.hpp"
#include "../../../gather/local.hpp"
#include "../../../histogram/local.hpp"
#include "../../../kernel.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../../map/source/upper.hpp"
#include "../../../numeric/source.hpp"
#include "../../../numeric/state.hpp"
#include "../../../partition/local.hpp"
#include "../../../range/local.hpp"
#include "../../../reduce/local.hpp"
#include "../../../scan/local.hpp"
#include "../../../scan/source.hpp"
#include "../../../scatter/local.hpp"
#include "../../../scatter/reduce/model.hpp"
#include "../../../segmented/local.hpp"
#include "../../../segmented/reduce/model.hpp"
#include "../../../sort/local/state.hpp"
#include "../../manifest.hpp"
#include "../../ops/prepare.hpp"
#include "../../pipeline/capacity.hpp"
#include "../../pipeline/evidence.hpp"
#include "../../pipeline/prepare/record.hpp"
#include "../../pipeline/recurrence.hpp"
#include "../../pipeline/source.hpp"
#include "../../pipeline/state.hpp"
#include "../../reset/source.hpp"
#include "../../window/source.hpp"

#include "../../../../primitive/block.hpp"
#include "../../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] static rund::AccelCheck PlanVulkanPipelineStructureCandidate(
    const rund::AccelContext &,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  const bool has_windows = reservation.window_count != 0u;
  if (reservation.backend_terminal_publication_count >
          reservation.backend_publication_count ||
      reservation.backend_window_descriptor_state_count >
          reservation.backend_window_state_count ||
      reservation.backend_indirect_dispatch_count >
          reservation.backend_window_dispatch_count ||
      (has_windows &&
       (reservation.backend_window_dispatch_count == 0u ||
        reservation.backend_window_state_count == 0u ||
        reservation.backend_window_descriptor_state_count == 0u)) ||
      (!has_windows &&
       (reservation.backend_window_dispatch_count != 0u ||
        reservation.backend_indirect_dispatch_count != 0u ||
        reservation.backend_window_state_count != 0u ||
        reservation.backend_window_descriptor_state_count != 0u))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.backend_query_count = 0u;
  if (!product(2u, reservation.window_count,
               reservation.backend_window_control_command_count) ||
      (reservation.backend_profile_command_count != 0u &&
       !product(2u, reservation.backend_profile_command_count,
                reservation.backend_query_count))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t commands = reservation.backend_dispatch_count;
  for (const std::uint64_t count :
       {reservation.backend_reset_dispatch_count,
        reservation.backend_status_command_count,
        reservation.backend_telemetry_command_count,
        reservation.backend_window_control_command_count,
        reservation.backend_indirect_dispatch_count,
        reservation.backend_publication_command_count, std::uint64_t{3u}}) {
    if (!add(commands, count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  std::uint64_t host = sizeof(VulkanPipeline);
  std::uint64_t record_host = 0u;
  if (!VulkanPipelineRecordHostBytes(reservation, record_host) ||
      !add(host, record_host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const bool selection_upper =
      !has_windows && reservation.backend_publication_count == 0u &&
      reservation.backend_profile_step_count == 0u &&
      reservation.nested_group_count == 0u &&
      reservation.map_recurrence.group_count == 0u &&
      reservation.map_recurrence.history_group_count == 0u &&
      reservation.authored_entry_count != 0u &&
      reservation.authored_entry_count <= PreparedPipelineStepCapacity;
  std::uint64_t selection_host = 0u;
  if (selection_upper &&
      (!product(reservation.authored_entry_count, sizeof(VulkanResidencyStep),
                selection_host) ||
       !add(selection_host, sizeof(VulkanResidencySelection)) ||
       !add(host, selection_host))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t bytes = 0u;
  std::uint64_t control_lease_count = 0u;
  std::uint64_t publication_descriptor_count = 0u;
  std::uint64_t window_descriptor_count = 0u;
  std::uint64_t control_binding_count = 0u;
  std::uint64_t publication_binding_count = 0u;
  std::uint64_t window_binding_count = 0u;
  std::uint64_t pipeline_descriptor_sets = 0u;
  std::uint64_t pipeline_descriptors = 0u;
  if (!rund::kernel::checked::add(reservation.backend_status_source_count, 1u,
                                  control_lease_count) ||
      !rund::kernel::checked::add(control_lease_count,
                                  reservation.backend_telemetry_count,
                                  control_lease_count) ||
      !rund::kernel::checked::add(
          reservation.backend_publication_count,
          reservation.backend_terminal_publication_count,
          publication_descriptor_count) ||
      !rund::kernel::checked::add(
          reservation.backend_window_descriptor_state_count,
          reservation.backend_indirect_dispatch_count,
          window_descriptor_count) ||
      !product(reservation.backend_status_source_count, 2u,
               control_binding_count) ||
      !rund::kernel::checked::add(control_binding_count, 2u,
                                  control_binding_count) ||
      !product(reservation.backend_telemetry_count,
               reservation.backend_profile_step_count == 0u ? 5u : 6u, bytes) ||
      !rund::kernel::checked::add(control_binding_count, bytes,
                                  control_binding_count) ||
      !product(publication_descriptor_count, 7u, publication_binding_count) ||
      !product(reservation.backend_window_descriptor_state_count, 8u,
               window_binding_count) ||
      !product(reservation.backend_indirect_dispatch_count, 3u, bytes) ||
      !rund::kernel::checked::add(window_binding_count, bytes,
                                  window_binding_count) ||
      !rund::kernel::checked::add(control_lease_count,
                                  publication_descriptor_count,
                                  pipeline_descriptor_sets) ||
      !rund::kernel::checked::add(pipeline_descriptor_sets,
                                  window_descriptor_count,
                                  pipeline_descriptor_sets) ||
      !rund::kernel::checked::add(control_binding_count,
                                  publication_binding_count,
                                  pipeline_descriptors) ||
      !rund::kernel::checked::add(pipeline_descriptors, window_binding_count,
                                  pipeline_descriptors)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t pipeline_source_bytes = 0u;
  std::uint64_t pipeline_source_storage = 0u;
  std::uint64_t pipeline_source_transient = 0u;
  std::uint64_t pipeline_source_dependency_count = 0u;
  const auto add_pipeline_source = [&](const std::uint64_t source_bytes) {
    std::uint64_t storage_bytes = 0u;
    return source_bytes != 0u &&
           backend_source_recipe::string_external_storage_upper_bytes(
               source_bytes, storage_bytes) &&
           add(pipeline_source_bytes, source_bytes) &&
           add(pipeline_source_storage, storage_bytes) &&
           add(pipeline_source_dependency_count, 1u) &&
           (pipeline_source_transient =
                std::max(pipeline_source_transient, storage_bytes),
            true);
  };
  const std::uint64_t telemetry_source_bytes =
      reservation.backend_profile_step_count == 0u
          ? VulkanTelemetrySourceText().size()
          : VulkanProfileSourceBytes();
  if (!add_pipeline_source(VulkanReduceStatusSourceText().size()) ||
      (reservation.backend_status_source_count != 0u &&
       !add_pipeline_source(VulkanCanonicalStatusSourceText().size())) ||
      (reservation.backend_telemetry_count != 0u &&
       !add_pipeline_source(telemetry_source_bytes)) ||
      (has_windows &&
       (!VulkanWindowSourceBytes(bytes) || !add_pipeline_source(bytes))) ||
      (reservation.backend_indirect_dispatch_count != 0u &&
       !add_pipeline_source(VulkanGateSourceText().size())) ||
      (reservation.backend_publication_count != 0u &&
       !add_pipeline_source(VulkanPublishSourceText().size()))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t pipeline_cache_host = 0u;
  std::uint64_t pipeline_cache_native = 0u;
  if (!product(pipeline_descriptor_sets,
               sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
               pipeline_cache_host) ||
      !product(pipeline_source_dependency_count, sizeof(VkDescriptorPool),
               bytes) ||
      !add(pipeline_cache_host, bytes) ||
      !product(pipeline_source_dependency_count,
               sizeof(VulkanCollectivePipeline), bytes) ||
      !add(pipeline_cache_host, bytes) ||
      !add(pipeline_cache_host, pipeline_source_storage) ||
      !product(pipeline_source_dependency_count, 4u, pipeline_cache_native) ||
      !add(pipeline_cache_native, pipeline_descriptor_sets) ||
      !add(reservation.template_source_bytes, pipeline_source_bytes) ||
      !add(reservation.template_host_bytes, pipeline_cache_host) ||
      !add(host, pipeline_cache_host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.source_transient_bytes =
      std::max(reservation.source_transient_bytes, pipeline_source_transient);
  if (reservation.backend_profile_step_count != 0u &&
      !add(host, sizeof(VulkanPipelineProfile))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!product(reservation.backend_telemetry_count,
               sizeof(VulkanPipelineTelemetryRecord), bytes) ||
      !add(host, bytes) ||
      !product(reservation.nested_group_count, sizeof(std::shared_ptr<void>),
               bytes) ||
      !add(host, bytes) ||
      !product(control_lease_count, sizeof(VulkanCollectiveDescriptorLease),
               bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_publication_count,
               sizeof(VulkanPipelinePublishRoute), bytes) ||
      !add(host, bytes) ||
      !product(publication_descriptor_count,
               sizeof(VulkanCollectiveDescriptorLease), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_window_dispatch_count,
               sizeof(VkDispatchIndirectCommand), bytes) ||
      !add(host, bytes) ||
      !product(reservation.window_count, sizeof(VulkanWindowRoute), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_indirect_dispatch_count,
               sizeof(VulkanGateRoute), bytes) ||
      !add(host, bytes) ||
      !product(window_descriptor_count, sizeof(VulkanCollectiveDescriptorLease),
               bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_profile_command_count,
               sizeof(std::uint8_t) + sizeof(std::uint32_t), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_query_count, sizeof(std::uint64_t), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_telemetry_command_count,
               reservation.backend_profile_step_count == 0u
                   ? sizeof(VulkanPipelineTelemetryParams)
                   : sizeof(VulkanPipelineProfileTelemetryParams),
               bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_window_control_command_count,
               sizeof(VulkanWindowParams), bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_indirect_dispatch_count,
               VulkanGateParameterBytes, bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !product(reservation.backend_publication_command_count,
               sizeof(VulkanPipelinePublishParams), bytes) ||
      !add(reservation.backend_parameter_bytes, bytes) ||
      !add(reservation.backend_parameter_bytes,
           2u * VulkanPipelineControlParameterBytes) ||
      !add(reservation.host_bytes, host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.backend_command_count = commands;
  reservation.backend_command_binding_count = pipeline_descriptors;
  if (!add(reservation.descriptor_set_count, pipeline_descriptor_sets) ||
      !add(reservation.descriptor_count, pipeline_descriptors)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Control owns summary unconditionally, arena only with status entries,
  // and profile only when requested. A live window owns four exact buffers:
  // states, mutable arguments, frozen original arguments, and owners.
  reservation.backend_native_buffer_count =
      1u +
      static_cast<std::uint64_t>(reservation.backend_status_entry_count != 0u) +
      static_cast<std::uint64_t>(reservation.backend_profile_step_count != 0u) +
      (has_windows ? 4u : 0u);
  reservation.backend_native_object_count = 3u;
  std::uint64_t selection_native_objects =
      selection_upper ? reservation.authored_entry_count : 0u;
  if (selection_upper &&
      (!add(selection_native_objects, 2u) ||
       !product(selection_native_objects, 3u, selection_native_objects))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!add(reservation.backend_native_object_count,
           reservation.backend_native_buffer_count) ||
      !add(reservation.backend_native_object_count, pipeline_cache_native) ||
      !add(reservation.backend_native_object_count, selection_native_objects) ||
      (reservation.backend_query_count != 0u &&
       !add(reservation.backend_native_object_count, 1u))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t native_payload = sizeof(PreparedPipelineControl);
  if (!product(reservation.backend_status_entry_count, sizeof(std::uint32_t),
               bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_profile_step_count,
               PreparedPipelineStepControlBytes, bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_state_count, sizeof(ResidentState),
               bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_dispatch_count,
               2u * sizeof(VkDispatchIndirectCommand), bytes) ||
      !add(native_payload, bytes) ||
      !product(reservation.backend_window_dispatch_count, sizeof(std::uint32_t),
               bytes) ||
      !add(native_payload, bytes) ||
      !add(reservation.native_bytes, native_payload) ||
      !add(reservation.native_bytes, reservation.backend_parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t descriptor_owner_bytes = 0u;
  std::uint64_t profile_description_bytes = 0u;
  if (!product(reservation.backend_window_state_count, sizeof(std::size_t),
               descriptor_owner_bytes) ||
      (reservation.backend_profile_step_count != 0u &&
       (!product(reservation.nested_group_count,
                 sizeof(VulkanPipelineWork) + sizeof(std::uint64_t),
                 profile_description_bytes))) ||
      !add(reservation.host_transient_bytes, descriptor_owner_bytes) ||
      !add(reservation.host_transient_bytes, profile_description_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)reservation;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
#endif
}

rund::AccelCheck PlanVulkanPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept {
  PreparedKernelPipelineReservation candidate = reservation;
  const rund::AccelCheck planned =
      PlanVulkanPipelineStructureCandidate(context, candidate);
  if (planned.ok) {
    reservation = candidate;
  }
  return planned;
}

} // namespace rund::node::accel::detail
