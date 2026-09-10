#include "../adapter/error.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/exception.hpp"
#include "../scratch.hpp"
#include "lease.hpp"
#include "prepared/local.hpp"

#include <limits>
#include <mutex>
#include <new>
#include <optional>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanResources(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const std::uint64_t dispatch_count,
    const KernelPreparationMode mode, const BoundResets *const resets,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    const KernelScratchLayout *const scratch,
    const BackendRun *const template_probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, std::shared_ptr<void> &prepared,
    PreparedMemory &memory) {
  memory = {};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  prepared.reset();
  if (steps == nullptr || step_count == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  try {
    VulkanKernelContext context{};
    const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    VulkanAdapter *const adapter = context.adapter;
    std::shared_ptr<VulkanKernelResources> resources{
        new VulkanKernelResources{}, DestroyPreparedVulkanKernelResources};
    resources->adapter = adapter;
    resources->dispatch_count = dispatch_count;
    resources->mode = mode;
    std::optional<VulkanScratch> scratch_arena;
    if (scratch != nullptr) {
      if (view_binds == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
      scratch_arena.emplace(pick, *scratch, *view_binds);
      if (!scratch_arena->valid()) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
    }
    const rund::AccelCheck view_ready =
        PrepareVulkanStepViews(pick, steps, step_count, mode, views, view_binds,
                               failed_node, *resources);
    if (!view_ready.ok) {
      return view_ready;
    }
    for (std::size_t index = 0u; index < resources->size(); ++index) {
      const VulkanKernelEntry *const entry = resources->entry(index);
      const std::uint64_t auxiliary =
          entry == nullptr ? 0u : VulkanViewDispatchCount(entry->view);
      if (auxiliary > std::numeric_limits<std::uint64_t>::max() -
                          resources->dispatch_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      resources->dispatch_count += auxiliary;
    }
    std::unique_lock<std::mutex> lock{adapter->mutex};
    if (!EnsureVulkanCommandResources(*adapter)) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
    const VulkanMemoryStats before = adapter->staging_memory;
    const rund::AccelCheck resets_ready = PrepareVulkanResets(
        *adapter, steps, resets, mode, failed_node, *resources);
    if (!resets_ready.ok) {
      return resets_ready;
    }
    const rund::AccelCheck template_ready = PrepareVulkanKernelProgramTemplate(
        pick, steps, step_count, mode, template_probe, templates, failed_node,
        *resources);
    if (!template_ready.ok) {
      return template_ready;
    }
    const rund::AccelCheck lease_storage = PrepareVulkanDescriptorLeaseStorage(
        steps, step_count, failed_node, *resources);
    if (!lease_storage.ok) {
      return lease_storage;
    }
    BeginVulkanCollectiveDescriptorEpoch(*adapter);
    rund::AccelCheck ready{};
    {
      VulkanLeaseScope lease_scope{*adapter, resources->descriptor_leases};
      try {
        VulkanScratchScope scratch_scope{
            scratch_arena.has_value() ? &scratch_arena.value() : nullptr};
        const rund::AccelCheck resets_ready =
            PrepareVulkanResetCommands(*adapter, *resources);
        ready = resets_ready.ok
                    ? PrepareVulkanSteps(pick, steps, step_count, mode,
                                         failed_node, *resources)
                    : resets_ready;
        for (const VulkanReset &clear : resources->resets) {
          if (ready.ok && clear.shader) {
            const std::uint64_t window =
                static_cast<std::uint64_t>(adapter->max_dispatch_groups) * 256u;
            const std::uint64_t commands =
                reset::Commands(clear.range.count(), window);
            if (commands > std::numeric_limits<std::uint64_t>::max() -
                               resources->dispatch_count) {
              ready = rund::AccelCheck{false, "accel_kernel_run_invalid"};
              break;
            }
            resources->dispatch_count += commands;
          }
        }
        if (ready.ok && !IsPipelinePrivatePreparation(mode)) {
          ready = RecordVulkanKernel(*adapter, *resources);
        }
      } catch (...) {
        backend_exception::RethrowUnlessCapacityException();
        RecordNode(failed_node, steps[0]);
        ready = rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    resources->shared_scratch =
        scratch_arena.has_value() && scratch_arena->used();
    if (ready.ok) {
      ProjectVulkanPreparedMemory(before, *adapter, *resources, memory);
    }
    lock.unlock();
    if (!ready.ok) {
      return ready;
    }
    prepared = std::move(resources);
    return rund::AccelCheck{true, "ok"};
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
#else
  (void)pick;
  (void)steps;
  (void)step_count;
  (void)dispatch_count;
  (void)mode;
  (void)resets;
  (void)views;
  (void)view_binds;
  (void)scratch;
  (void)template_probe;
  (void)templates;
  (void)failed_node;
  (void)prepared;
  (void)memory;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
