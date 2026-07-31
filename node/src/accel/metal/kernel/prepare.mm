#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../scratch.hpp"
#include "local.hpp"
#include "ops/table.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck PrepareMetalStep(const rund::AccelDevice &pick,
                                  const BoundStep &step,
                                  const MetalKernelOps &ops,
                                  std::shared_ptr<void> &resources) {
  if (ops.prepare != nullptr) {
    return ops.prepare(pick, step, resources);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck PrepareMetalSteps(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const KernelPreparationMode mode,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    std::uint32_t *const failed_node, MetalKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || !resources.reserve(step_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const KernelPreparationScope preparation{mode};
  bool scratch_seen = false;
  for (std::size_t index = 0u; index < step_count; ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    entry->ops = MetalKernelOpsFor(steps[index].step->kind());
    entry->resets = steps[index].resets;
    entry->barrier_before = steps[index].barrier_before;
    const rund::AccelCheck view = PrepareMetalViewLowering(
        pick, steps[index], mode, views, view_binds, entry->view);
    if (!view.ok) {
      RecordNode(failed_node, steps[index]);
      return view;
    }
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    MetalScratch *const scratch = ActiveMetalScratch();
    if (scratch != nullptr) {
      scratch->reset();
    }
    const rund::AccelCheck prepare =
        PrepareMetalStep(pick, prepared_step, entry->ops, entry->resource);
    if (!prepare.ok) {
      RecordNode(failed_node, steps[index]);
      return prepare;
    }
    const bool scratch_used = scratch != nullptr && scratch->active();
    entry->barrier_before =
        entry->barrier_before || (scratch_seen && scratch_used);
    scratch_seen = scratch_seen || scratch_used;
    if (entry->ops.memory != nullptr) {
      accumulate_memory(
          resources.memory,
          entry->ops.memory(entry->resource, pick.caps.staging_bytes));
    }
    std::uint64_t traffic = 0u;
    accumulate_memory(
        resources.memory,
        MetalViewMemory(entry->view, pick.caps.staging_bytes, traffic));
    resources.traffic =
        ::rund::detail::counter::SaturatingAdd(resources.traffic, traffic);
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
