#include "local.hpp"

#include "../../pipeline/named.hpp"
#include "../../scratch.hpp"

#include <kernel/core/checked.hpp>

#include <limits>
#include <new>
#include <optional>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck PrepareMetalResources(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const std::uint64_t dispatch_count,
    const KernelPreparationMode mode, const BoundResets *const resets,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    const KernelScratchLayout *const scratch,
    const BackendRun *const template_probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, std::shared_ptr<void> &prepared,
    PreparedMemory &memory) {
  @autoreleasepool {
    prepared.reset();
    memory = {};
    if (steps == nullptr || step_count == 0u) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid = ValidateMetalKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    std::shared_ptr<MetalKernelResources> resources{};
    try {
      resources = std::make_shared<MetalKernelResources>();
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    std::optional<MetalScratch> scratch_arena;
    try {
      if (scratch != nullptr) {
        if (view_binds == nullptr) {
          return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
        }
        scratch_arena.emplace(pick, *scratch, *view_binds);
        if (!scratch_arena->valid()) {
          return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
        }
      }
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    rund::AccelCheck ready{};
    try {
      MetalScratchScope scratch_scope{
          scratch_arena.has_value() ? &scratch_arena.value() : nullptr};
      ready =
          PrepareMetalSteps(pick, steps, step_count, mode, views, view_binds,
                            template_probe, templates, failed_node, *resources);
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      return ready;
    }
    bool resets_ready = false;
    try {
      resets_ready =
          PrepareMetalResets(pick, *context.adapter, resets, *resources);
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!resets_ready) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    resources->adapter = context.adapter;
    std::uint64_t physical_dispatch_count = dispatch_count;
    for (std::size_t index = 0u; index < resources->size(); ++index) {
      const MetalKernelEntry *const entry = resources->entry(index);
      const std::uint64_t auxiliary =
          entry == nullptr ? 0u : MetalViewDispatchCount(entry->view);
      if (auxiliary >
          std::numeric_limits<std::uint64_t>::max() - physical_dispatch_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      physical_dispatch_count += auxiliary;
    }
    if (resources->reset_count >
        std::numeric_limits<std::uint64_t>::max() - physical_dispatch_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    physical_dispatch_count += resources->reset_count;
    resources->dispatch_count = physical_dispatch_count;
    resources->mode = mode;
    resources->shared_scratch =
        scratch_arena.has_value() && scratch_arena->used();
    memory = resources->memory;
    prepared = std::move(resources);
    return rund::AccelCheck{true, "ok"};
  }
}
#else
rund::AccelCheck
PrepareMetalResources(const rund::AccelDevice &, const BoundStep *, std::size_t,
                      std::uint64_t, KernelPreparationMode, const BoundResets *,
                      const KernelViewLayout *, const RunBinds *,
                      const KernelScratchLayout *, const BackendRun *,
                      PreparedKernelTemplateRegistry *, std::uint32_t *,
                      std::shared_ptr<void> &, PreparedMemory &memory) {
  memory = {};
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}
#endif

} // namespace rund::node::accel::detail
