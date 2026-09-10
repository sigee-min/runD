#include "local.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck PrepareVulkanDescriptorLeaseStorage(
    const BoundStep *const steps, const std::size_t step_count,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || resources.size() != step_count ||
      !resources.descriptor_leases.empty()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t lease_count = 0u;
  const bool captured = IsPipelinePrivatePreparation(resources.mode);
  for (const VulkanReset &clear : resources.resets) {
    if (clear.shader &&
        !rund::kernel::checked::add(lease_count, 1u, lease_count)) {
      RecordNode(failed_node, steps[0]);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr ||
        !rund::kernel::checked::add(
            lease_count, VulkanViewDispatchCount(entry->view), lease_count)) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (captured && (resources.program == nullptr ||
                   resources.program->steps.size() != step_count)) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    if (prepared_step.step == nullptr || prepared_step.planned == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const PreparedBackendManifest manifest =
        captured ? resources.program->steps[index].manifest
                 : BuildVulkanBackendManifest(
                       *prepared_step.step, prepared_step.planned->plan,
                       &prepared_step,
                       resources.adapter == nullptr
                           ? 0u
                           : resources.adapter->max_dispatch_groups);
    if (!manifest.ok) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, manifest.reason};
    }
    if (!rund::kernel::checked::add(
            lease_count, manifest.descriptor_lease_count, lease_count)) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (lease_count > std::numeric_limits<std::size_t>::max()) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.descriptor_leases.reserve(static_cast<std::size_t>(lease_count));
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
