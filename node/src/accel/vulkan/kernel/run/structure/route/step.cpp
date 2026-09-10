#include "local.hpp"

#include "../../../../../kernel/backend/template/arithmetic.hpp"
#include "../../../../../kernel/status.hpp"

#include "../../../../collective/chunk.hpp"
#include "../../../../compact/local.hpp"
#include "../../../../descriptor.hpp"
#include "../../../../gather/local.hpp"
#include "../../../../histogram/local.hpp"
#include "../../../../kernel.hpp"
#include "../../../../map/api.hpp"
#include "../../../../map/local.hpp"
#include "../../../../numeric/state.hpp"
#include "../../../../partition/local.hpp"
#include "../../../../range/local.hpp"
#include "../../../../reduce/local.hpp"
#include "../../../../scan/local.hpp"
#include "../../../../scatter/local.hpp"
#include "../../../../scatter/reduce/model.hpp"
#include "../../../../segmented/local.hpp"
#include "../../../../segmented/reduce/model.hpp"
#include "../../../../sort/local/state.hpp"
#include "../../../manifest.hpp"
#include "../../../ops/prepare.hpp"
#include "../../../pipeline/capacity.hpp"
#include "../../../pipeline/evidence.hpp"
#include "../../../pipeline/state.hpp"

#include "../../../../../primitive/block.hpp"
#include "../../../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include "../../storage.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] rund::AccelCheck
PlanVulkanStepStructure(const KernelExecutionStep &step,
                        const rund::kernel::ComputePlan &plan,
                        const BoundStep *const bound, const KernelViewLayout *,
                        const std::uint64_t max_dispatch_groups,
                        PreparedKernelRouteReservation &reservation) noexcept {
  const PreparedBackendManifest manifest =
      BuildVulkanBackendManifest(step, plan, bound, max_dispatch_groups);
  if (!manifest.ok) {
    return rund::AccelCheck{false, manifest.reason};
  }
  // The shared planner already charged plan.dispatch_count. Vulkan's capture
  // manifest is the exact physical primitive command authority, so retain only
  // its additional stages here; reset and View commands have separate owners.
  std::uint64_t primitive_dispatch_count = 0u;
  if (!rund::kernel::checked::add(manifest.capture_direct_dispatch_count,
                                  manifest.capture_indirect_dispatch_count,
                                  primitive_dispatch_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (primitive_dispatch_count >= plan.dispatch_count) {
    if (!backend_template_plan::add(reservation.dispatch_count,
                                    primitive_dispatch_count -
                                        plan.dispatch_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } else {
    const std::uint64_t removed =
        plan.dispatch_count - primitive_dispatch_count;
    if (removed > reservation.dispatch_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    reservation.dispatch_count -= removed;
  }
  std::uint64_t route = 0u;
  std::uint64_t map_descriptor_sets = 0u;
  std::uint64_t map_check_binding_count = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const std::uint64_t route_dispatches =
        VulkanMapRouteDispatches(plan, bound);
    const std::uint64_t check_count = UniqueVulkanMapCheckCount(step.artifact);
    route = sizeof(VulkanMapEncodeResources);
    if (!backend_template_plan::add(reservation.template_host_bytes,
                                    sizeof(VulkanMapTemplateResources)) ||
        !backend_template_plan::add(reservation.template_host_bytes,
                                    sizeof(VulkanMapDescriptorArena)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.input_buffer_count, sizeof(InputWindowPlan)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.input_buffer_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes,
                            plan.output_buffer_count,
                            sizeof(VulkanMapBindingLayout)) ||
        !AddVulkanHostBytes(reservation.template_host_bytes, check_count,
                            sizeof(VulkanMapCheck)) ||
        !AddVulkanHostBytes(route, plan.input_buffer_count,
                            sizeof(VulkanResidentBufferResult)) ||
        !AddVulkanHostBytes(route, plan.output_buffer_count,
                            sizeof(VulkanResidentBufferResult)) ||
        !AddVulkanHostBytes(route, check_count, sizeof(std::uint64_t)) ||
        !AddVulkanHostBytes(route, route_dispatches, sizeof(VkDescriptorSet))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (check_count != 0u &&
        !rund::kernel::checked::add(check_count, 3u, map_check_binding_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    map_descriptor_sets = route_dispatches;
    break;
  }
  case rund::kernel::NodeKind::Scan:
    route =
        sizeof(VulkanKernelScanResources) + sizeof(VulkanScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    route = sizeof(VulkanSegmentedScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    route = sizeof(VulkanSegmentedReduceResources);
    break;
  case rund::kernel::NodeKind::Sort:
    route = sizeof(VulkanSortEncodeResources);
    break;
  case rund::kernel::NodeKind::Compact:
    route = sizeof(VulkanCompactEncodeResources);
    break;
  case rund::kernel::NodeKind::Gather:
    route = sizeof(VulkanGatherEncodeResources);
    break;
  case rund::kernel::NodeKind::Histogram:
    route = sizeof(VulkanHistogramEncodeResources);
    break;
  case rund::kernel::NodeKind::Partition:
    route = sizeof(VulkanPartitionEncodeResources) +
            sizeof(VulkanScanEncodeResources);
    break;
  case rund::kernel::NodeKind::Reduce:
    route = sizeof(VulkanReduceEncodeResources);
    if (!AddVulkanHostBytes(
            route, step.operation.get<operation::Reduce>().plan.pass_count,
            sizeof(VkDescriptorSet))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    break;
  case rund::kernel::NodeKind::Scatter:
    route = sizeof(VulkanScatterEncodeResources);
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    route = sizeof(VulkanScatterReduceResources);
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
    route = sizeof(VulkanRangeResources);
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    route = sizeof(VulkanNumericPrepared);
    break;
  }
  const std::uint64_t immutable_host_bytes =
      step.kind() == rund::kernel::NodeKind::Map
          ? 0u
          : sizeof(VulkanKernelImmutablePipelines);
  std::uint64_t cache_owner_host_bytes = 0u;
  if (step.kind() == rund::kernel::NodeKind::Map) {
    if (manifest.native_pipeline_dependency_count == 0u ||
        !AddVulkanHostBytes(cache_owner_host_bytes,
                            manifest.native_pipeline_dependency_count - 1u,
                            sizeof(VulkanCollectivePipeline)) ||
        !backend_template_plan::add(cache_owner_host_bytes,
                                    sizeof(VulkanCachedPipeline))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } else if (!AddVulkanHostBytes(cache_owner_host_bytes,
                                 manifest.native_pipeline_dependency_count,
                                 sizeof(VulkanCollectivePipeline))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t collective_descriptor_sets = 0u;
  if (map_descriptor_sets > manifest.descriptor_set_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  collective_descriptor_sets =
      manifest.descriptor_set_count - map_descriptor_sets;
  reservation.source_transient_bytes = std::max(
      reservation.source_transient_bytes, manifest.cold_source_transient_bytes);
  return AddVulkanHostBytes(reservation.host_transient_bytes,
                            manifest.status_source_count,
                            sizeof(VulkanPipelineCanonicalStatus)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes,
                                    map_check_binding_count,
                                    sizeof(VulkanStorageBinding)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes,
                                    manifest.descriptor_dependency_count,
                                    sizeof(VulkanKernelDescriptorDependency)) &&
                 AddVulkanHostBytes(reservation.host_transient_bytes, 2u,
                                    sizeof(PreparedProgramStatusSlice)) &&
                 backend_template_plan::add(reservation.route_host_bytes,
                                            route) &&
                 AddVulkanHostBytes(
                     reservation.route_host_bytes, collective_descriptor_sets,
                     sizeof(VkDescriptorSet) + sizeof(std::uint8_t)) &&
                 AddVulkanHostBytes(reservation.route_host_bytes,
                                    manifest.descriptor_lease_count,
                                    sizeof(VulkanCollectiveDescriptorLease)) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            immutable_host_bytes) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            cache_owner_host_bytes) &&
                 AddVulkanHostBytes(reservation.template_host_bytes,
                                    manifest.descriptor_dependency_count,
                                    sizeof(VulkanKernelDescriptorDependency)) &&
                 AddVulkanHostBytes(
                     reservation.template_host_bytes,
                     manifest.native_pipeline_dependency_count -
                         static_cast<std::uint64_t>(
                             step.kind() == rund::kernel::NodeKind::Map),
                     sizeof(VkDescriptorPool)) &&
                 backend_template_plan::add(reservation.template_source_bytes,
                                            manifest.cold_cache_source_bytes) &&
                 backend_template_plan::add(
                     reservation.template_host_bytes,
                     manifest.cold_cache_source_storage_bytes) &&
                 backend_template_plan::add(
                     reservation.template_native_allocation_count,
                     manifest.cold_cache_native_object_count) &&
                 backend_template_plan::add(reservation.descriptor_set_count,
                                            manifest.descriptor_set_count) &&
                 backend_template_plan::add(
                     reservation.descriptor_count,
                     manifest.descriptor_binding_count) &&
                 backend_template_plan::add(
                     reservation.capture_direct_dispatch_count,
                     manifest.capture_direct_dispatch_count) &&
                 backend_template_plan::add(
                     reservation.capture_indirect_dispatch_count,
                     manifest.capture_indirect_dispatch_count) &&
                 backend_template_plan::add(reservation.status_source_count,
                                            manifest.status_source_count) &&
                 backend_template_plan::add(reservation.status_entry_count,
                                            manifest.status_entry_count) &&
                 backend_template_plan::add(reservation.status_command_count,
                                            manifest.status_command_count) &&
                 backend_template_plan::add(reservation.status_parameter_bytes,
                                            manifest.status_parameter_bytes) &&
                 backend_template_plan::add(reservation.telemetry_source_count,
                                            manifest.telemetry_source_count)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}


#endif

} // namespace rund::node::accel::detail
