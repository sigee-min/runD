#include "local.hpp"
#include "../../../../../kernel/backend/source/storage.hpp"

#include "../../../../../kernel/backend/template/arithmetic.hpp"
#include "../../../../../kernel/backend/template/reservation.hpp"
#include "../../../../../kernel/status.hpp"

#include "../../../../descriptor.hpp"
#include "../../../../map/source/upper.hpp"
#include "../../../pipeline/capacity.hpp"
#include "../../../pipeline/source.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../reset/source.hpp"

#include "../../storage.hpp"

#include <algorithm>
#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
PlanVulkanViewCaptureCount(const KernelViewLayout *const views,
                           const VulkanAdapter &adapter,
                           std::uint64_t &count) noexcept {
  count = 0u;
  if (views == nullptr) {
    return true;
  }
  for (const KernelViewSlot &view : *views) {
    const rund::kernel::ResidentBufferRef ref{
        .bytes = view.backing_bytes,
        .offset_bytes = view.offset_bytes,
        .element_bytes = view.element_bytes,
        .stride_bytes = view.stride_bytes,
        .count = view.count,
        .usage = view.usage,
    };
    std::uint64_t begin = 0u;
    while (begin < ref.count) {
      StorageRange range{};
      if (!PlanStoragePage(adapter, ref, begin, range) || range.count == 0u ||
          !rund::kernel::checked::add(begin, range.count, begin) ||
          !rund::kernel::checked::add(count, 1u, count)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

[[nodiscard]] rund::AccelCheck CompleteVulkanRouteCaptureStructure(
    const KernelViewLayout *const views, const std::uint64_t reset_count,
    const VulkanAdapter &adapter,
    PreparedKernelRouteReservation &reservation) noexcept {
  std::uint64_t view_dispatch_count = 0u;
  std::uint64_t auxiliary_set_count = 0u;
  std::uint64_t auxiliary_binding_count = 0u;
  std::uint64_t auxiliary_dependency_count = 0u;
  std::uint64_t lease_bytes = 0u;
  std::uint64_t reset_source_storage = 0u;
  std::uint64_t reset_route_descriptor_host = 0u;
  std::uint64_t reset_template_descriptor_host = 0u;
  std::uint64_t reset_native_objects = 0u;
  std::uint64_t view_source_storage = 0u;
  std::uint64_t view_route_descriptor_host = 0u;
  std::uint64_t view_template_descriptor_host = 0u;
  std::uint64_t view_native_objects = 0u;
  if (!PlanVulkanViewCaptureCount(views, adapter, view_dispatch_count) ||
      !rund::kernel::checked::add(reset_count, view_dispatch_count,
                                  auxiliary_set_count) ||
      !rund::kernel::checked::mul(view_dispatch_count, 2u,
                                  auxiliary_binding_count) ||
      !rund::kernel::checked::add(auxiliary_binding_count, reset_count,
                                  auxiliary_binding_count) ||
      !rund::kernel::checked::add(reset_count == 0u ? 0u : 1u,
                                  view_dispatch_count == 0u ? 0u : 1u,
                                  auxiliary_dependency_count) ||
      !rund::kernel::checked::mul(auxiliary_set_count,
                                  sizeof(VulkanCollectiveDescriptorLease),
                                  lease_bytes) ||
      (reset_count != 0u &&
       (!backend_source_recipe::string_external_storage_upper_bytes(
            VulkanResetSourceText().size(), reset_source_storage) ||
        !rund::kernel::checked::mul(
            reset_count, sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
            reset_route_descriptor_host) ||
        !rund::kernel::checked::add(reset_template_descriptor_host,
                                    sizeof(VulkanKernelDescriptorDependency) +
                                        sizeof(VkDescriptorPool) +
                                        sizeof(VulkanCollectivePipeline),
                                    reset_template_descriptor_host) ||
        !rund::kernel::checked::add(reset_count, 4u, reset_native_objects))) ||
      (view_dispatch_count != 0u &&
       (!backend_source_recipe::string_external_storage_upper_bytes(
            VulkanViewSourceText().size(), view_source_storage) ||
        !rund::kernel::checked::mul(
            view_dispatch_count, sizeof(VkDescriptorSet) + sizeof(std::uint8_t),
            view_route_descriptor_host) ||
        !rund::kernel::checked::add(view_template_descriptor_host,
                                    sizeof(VulkanKernelDescriptorDependency) +
                                        sizeof(VkDescriptorPool) +
                                        sizeof(VulkanCollectivePipeline),
                                    view_template_descriptor_host) ||
        !rund::kernel::checked::add(view_dispatch_count, 4u,
                                    view_native_objects))) ||
      !backend_template_plan::add(reservation.capture_direct_dispatch_count,
                                  reservation.reset_dispatch_count) ||
      !backend_template_plan::add(reservation.capture_direct_dispatch_count,
                                  view_dispatch_count) ||
      // View commands are encoded body work. Reset remains exclusively in
      // reset_dispatch_count even though both participate in capture gating.
      !backend_template_plan::add(reservation.dispatch_count,
                                  view_dispatch_count) ||
      !backend_template_plan::add(reservation.route_host_bytes, lease_bytes) ||
      !backend_template_plan::add(reservation.route_host_bytes,
                                  reset_route_descriptor_host) ||
      !backend_template_plan::add(reservation.route_host_bytes,
                                  view_route_descriptor_host) ||
      !AddVulkanHostBytes(reservation.host_transient_bytes,
                          auxiliary_dependency_count,
                          sizeof(VulkanKernelDescriptorDependency)) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  reset_template_descriptor_host) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  reset_source_storage) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  view_template_descriptor_host) ||
      !backend_template_plan::add(reservation.template_host_bytes,
                                  view_source_storage) ||
      !backend_template_plan::add(
          reservation.template_source_bytes,
          reset_count == 0u ? 0u : VulkanResetSourceText().size()) ||
      !backend_template_plan::add(
          reservation.template_source_bytes,
          view_dispatch_count == 0u ? 0u : VulkanViewSourceText().size()) ||
      !backend_template_plan::add(reservation.template_native_allocation_count,
                                  reset_native_objects) ||
      !backend_template_plan::add(reservation.template_native_allocation_count,
                                  view_native_objects) ||
      !backend_template_plan::add(reservation.descriptor_set_count,
                                  auxiliary_set_count) ||
      !backend_template_plan::add(reservation.descriptor_count,
                                  auxiliary_binding_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.source_transient_bytes =
      std::max(reservation.source_transient_bytes,
               std::max(reset_source_storage, view_source_storage));
  return rund::AccelCheck{true, "ok"};
}


#endif

} // namespace rund::node::accel::detail
