#include "capacity.hpp"

#include "../../../../kernel/backend/exception.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck DescribeVulkanPipelineCapacity(
    const std::span<const BackendBatchEntry> templates,
    VulkanPipelineDescriptionCapacity &capacity) noexcept {
  capacity = {};
  for (const BackendBatchEntry &entry : templates) {
    const auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<const VulkanKernelResources *>(entry.prepared->get());
    if (resources == nullptr || resources->program == nullptr ||
        resources->program->steps.size() != resources->size() ||
        !IsPipelinePrivatePreparation(resources->mode) ||
        !rund::kernel::checked::add(capacity.step_count, resources->size(),
                                    capacity.step_count)) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    for (const VulkanKernelProgramStepTemplate &step :
         resources->program->steps) {
      if (!step.manifest.ok ||
          !rund::kernel::checked::add(capacity.status_source_count,
                                      step.manifest.status_source_count,
                                      capacity.status_source_count) ||
          !rund::kernel::checked::add(capacity.status_entry_count,
                                      step.manifest.status_entry_count,
                                      capacity.status_entry_count) ||
          !rund::kernel::checked::add(capacity.telemetry_source_count,
                                      step.manifest.telemetry_source_count,
                                      capacity.telemetry_source_count)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }
  if (capacity.step_count > std::numeric_limits<std::size_t>::max() ||
      capacity.status_source_count > std::numeric_limits<std::size_t>::max() ||
      capacity.status_entry_count > std::numeric_limits<std::size_t>::max() ||
      capacity.telemetry_source_count >
          std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
