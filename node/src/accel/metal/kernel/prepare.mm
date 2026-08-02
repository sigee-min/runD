#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/prepared/template_registry.hpp"
#include "../compact/local.hpp"
#include "../gather/local.hpp"
#include "../histogram/local.hpp"
#include "../numeric/state.hpp"
#include "../partition/local.hpp"
#include "../reduce/local.hpp"
#include "../scan/kernel/local.hpp"
#include "../scatter/local.hpp"
#include "../scatter/reduce/model.hpp"
#include "../scratch.hpp"
#include "../segmented/local.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/local.hpp"
#include "../stencil/local.hpp"
#include "local.hpp"
#include "ops/table.hpp"

#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] constexpr bool
SameMetalTemplateRouteDemand(const BackendTemplateRouteDemand left,
                             const BackendTemplateRouteDemand right) noexcept {
  return left.owner_count == right.owner_count &&
         left.route_copies == right.route_copies &&
         left.capacity == right.capacity;
}

[[nodiscard]] bool MatchMetalProgramTemplate(const void *const prepared,
                                             const void *const probe) noexcept {
  if (prepared == nullptr ||
      MetalKernelTemplateKindOf(prepared) != MetalKernelTemplateKind::Program) {
    return false;
  }
  const auto *const program =
      static_cast<const MetalKernelProgramTemplate *>(prepared);
  const auto *const run = static_cast<const BackendRun *>(probe);
  return program != nullptr && program->signature != nullptr &&
         run != nullptr && program->route_demand.valid() &&
         run->template_route_demand.valid() &&
         SameMetalTemplateRouteDemand(program->route_demand,
                                      run->template_route_demand) &&
         backend_template_plan::same_template(*program->signature, *run, 1u);
}

[[nodiscard]] rund::AccelCheck PrepareMetalMapTemplateStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned == nullptr ||
      step.planned->artifact == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalMapTemplate(
      pick, step.planned->plan, *step.planned->artifact,
      step.map_windows.data(), step.map_windows.size(), MapBindingFor(step),
      step.control, prepared);
}

[[nodiscard]] rund::AccelCheck PrepareMetalMapRouteStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources) {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned == nullptr ||
      step.planned->artifact == nullptr || prepared == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalMapRoute(pick, step.planned->plan, *step.planned->artifact,
                              step.map_windows.data(), step.map_windows.size(),
                              MapBindingFor(step), step.control,
                              std::move(prepared), resources);
}

template <typename Resource>
[[nodiscard]] const Resource *
MetalPrimitiveResource(const std::shared_ptr<void> &resource) noexcept {
  return static_cast<const Resource *>(resource.get());
}

[[nodiscard]] rund::AccelCheck FreezeMetalPrimitivePipelines(
    const rund::kernel::NodeKind kind, const std::shared_ptr<void> &resource,
    std::shared_ptr<const MetalKernelImmutablePipelines> &out) {
  std::shared_ptr<MetalKernelImmutablePipelines> frozen;
  try {
    frozen = std::make_shared<MetalKernelImmutablePipelines>();
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_capacity"};
  }
  switch (kind) {
  case rund::kernel::NodeKind::Scan: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScanEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->block, raw->prefix, raw->offset};
    frozen->count = 3u;
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSegmentedScanEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->block, raw->prefix, raw->offset};
    frozen->count = 3u;
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSegmentedReduceResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.classify, raw->pipelines.prefix,
                      raw->pipelines.scatter, raw->pipelines.reduce};
    frozen->count = 4u;
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto *const raw =
        MetalPrimitiveResource<MetalSortEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.dispatch, raw->pipelines.histogram,
                      raw->pipelines.prefix, raw->pipelines.base,
                      raw->pipelines.scatter};
    frozen->count = 5u;
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    const auto *const raw =
        MetalPrimitiveResource<MetalCompactEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages =
        raw->block_offset_path
            ? std::array<std::shared_ptr<void>,
                         5u>{raw->pipelines.count_blocks,
                             raw->pipelines.scatter_blocks, raw->scan_block,
                             raw->scan_prefix, raw->scan_offset}
            : std::array<std::shared_ptr<void>, 5u>{
                  raw->pipelines.scatter, raw->pipelines.status,
                  raw->scan_block, raw->scan_prefix, raw->scan_offset};
    frozen->count = 5u;
    break;
  }
  case rund::kernel::NodeKind::Gather: {
    const auto *const raw =
        MetalPrimitiveResource<MetalGatherEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->control_pipeline, raw->gather_pipeline};
    frozen->count = 2u;
    break;
  }
  case rund::kernel::NodeKind::Histogram: {
    const auto *const raw =
        MetalPrimitiveResource<MetalHistogramEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.clear, raw->pipelines.count};
    frozen->count = 2u;
    break;
  }
  case rund::kernel::NodeKind::Partition: {
    const auto *const raw =
        MetalPrimitiveResource<MetalPartitionEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->pipelines.classify, raw->pipelines.scatter,
                      raw->scan_block, raw->scan_prefix, raw->scan_offset};
    frozen->count = 5u;
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalReduceEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::Scatter: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScatterEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    const auto *const raw =
        MetalPrimitiveResource<MetalScatterReduceResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages = {raw->control_pipeline, raw->init_pipeline,
                      raw->fold_pipeline};
    frozen->count = 3u;
    break;
  }
  case rund::kernel::NodeKind::Stencil: {
    const auto *const raw =
        MetalPrimitiveResource<MetalStencilEncodeResources>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum: {
    const auto *const raw =
        MetalPrimitiveResource<MetalNumericPrepared>(resource);
    if (raw == nullptr) {
      break;
    }
    frozen->stages[0u] = raw->pipeline;
    frozen->count = 1u;
    break;
  }
  case rund::kernel::NodeKind::Map:
    return {false, "accel_kernel_template_invalid"};
  }
  if (!frozen->ready(frozen->count)) {
    return {false, "accel_metal_pipeline_unavailable"};
  }
  out = std::move(frozen);
  return {true, "ok"};
}

[[nodiscard]] rund::AccelCheck AcquireMetalProgramTemplate(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const BackendRun *const probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, MetalKernelResources &resources,
    bool &publish_required) {
  publish_required = false;
  if (probe == nullptr || probe->steps == nullptr ||
      probe->step_count != step_count || probe->steps[0].step == nullptr ||
      !probe->template_route_demand.valid()) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const KernelExecutionStep *const authority = probe->steps[0].step;
  const std::uint64_t variant_hi = probe->execution == nullptr
                                       ? step_count
                                       : probe->execution->admission.kernel_id;
  const std::uint64_t variant_lo =
      (static_cast<std::uint64_t>(step_count) << 32u) ^
      probe->original_dispatch_count ^ probe->final_dispatch_count;
  if (templates != nullptr) {
    std::shared_ptr<void> found = FindPreparedKernelTemplate(
        *templates, authority, variant_hi, variant_lo,
        MatchMetalProgramTemplate, probe);
    if (found != nullptr) {
      resources.program =
          std::static_pointer_cast<MetalKernelProgramTemplate>(found);
      return rund::AccelCheck{true, "ok"};
    }
  }

  std::shared_ptr<MetalKernelProgramTemplate> program;
  try {
    program = std::make_shared<MetalKernelProgramTemplate>();
    program->signature = probe;
    program->route_demand = probe->template_route_demand;
    program->steps.reserve(step_count);
    if (program->steps.capacity() != step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    program->steps.resize(step_count);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    MetalKernelProgramStepTemplate &template_step = program->steps[index];
    template_step.ops = MetalKernelOpsFor(steps[index].step->kind());
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    if (prepared_step.step == nullptr || prepared_step.planned == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    template_step.manifest = BuildMetalBackendManifest(
        *prepared_step.step, prepared_step.planned->plan, &prepared_step, 1u);
    if (!template_step.manifest.ok) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, template_step.manifest.reason};
    }
    if (steps[index].step->kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    std::shared_ptr<const MetalMapTemplateResources> prepared_map;
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalMapTemplateStep(pick, prepared_step, prepared_map);
    } catch (const std::bad_alloc &) {
      ready = rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      RecordNode(failed_node, steps[index]);
      return ready;
    }
    template_step.immutable = std::move(prepared_map);
  }

  resources.program = std::move(program);
  publish_required = true;
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
PublishMetalProgramTemplate(const BackendRun &probe,
                            PreparedKernelTemplateRegistry *const templates,
                            MetalKernelResources &resources) {
  if (resources.program == nullptr || probe.steps == nullptr ||
      probe.step_count == 0u || probe.steps[0u].step == nullptr ||
      probe.ops == nullptr) {
    return {false, "accel_kernel_template_invalid"};
  }
  if (templates == nullptr) {
    return {true, "ok"};
  }
  const std::uint64_t variant_hi = probe.execution == nullptr
                                       ? probe.step_count
                                       : probe.execution->admission.kernel_id;
  const std::uint64_t variant_lo =
      (static_cast<std::uint64_t>(probe.step_count) << 32u) ^
      probe.original_dispatch_count ^ probe.final_dispatch_count;
  std::shared_ptr<void> published = resources.program;
  const rund::AccelCheck stored = PublishPreparedKernelTemplate(
      *templates, probe.steps[0u].step, variant_hi, variant_lo,
      *probe.ops, MatchMetalProgramTemplate, &probe, published);
  if (!stored.ok) {
    return stored;
  }
  resources.program =
      std::static_pointer_cast<MetalKernelProgramTemplate>(published);
  return {true, "ok"};
}

} // namespace

rund::AccelCheck
PrepareMetalStep(const rund::AccelDevice &pick, const BoundStep &step,
                 const MetalKernelOps &ops,
                 const MetalKernelImmutablePipelines *pipelines,
                 std::shared_ptr<void> &resources) {
  if (ops.prepare != nullptr) {
    return ops.prepare(pick, step, pipelines, resources);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck PrepareMetalSteps(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const KernelPreparationMode mode,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    const BackendRun *const template_probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, MetalKernelResources &resources) {
  if (steps == nullptr || step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  try {
    if (!resources.reserve(step_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const KernelPreparationScope preparation{mode};
  for (std::size_t index = 0u; index < step_count; ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    entry->resets = steps[index].resets;
    entry->barrier_before = steps[index].barrier_before;
    rund::AccelCheck view{};
    try {
      view = PrepareMetalViewLowering(pick, steps[index], mode, views,
                                      view_binds, entry->view);
    } catch (const std::bad_alloc &) {
      view = rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!view.ok) {
      RecordNode(failed_node, steps[index]);
      return view;
    }
  }

  bool publish_required = false;
  const rund::AccelCheck template_ready =
      IsPipelinePrivatePreparation(mode)
          ? AcquireMetalProgramTemplate(pick, steps, step_count, template_probe,
                                        templates, failed_node, resources,
                                        publish_required)
          : rund::AccelCheck{true, "ok"};
  if (!template_ready.ok) {
    return template_ready;
  }
  if (resources.program != nullptr &&
      resources.program->steps.size() != step_count) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }

  bool scratch_seen = false;
  for (std::size_t index = 0u; index < step_count; ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    entry->ops = resources.program == nullptr
                     ? MetalKernelOpsFor(steps[index].step->kind())
                     : resources.program->steps[index].ops;
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    if (prepared_step.step == nullptr || prepared_step.planned == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    MetalScratch *const scratch = ActiveMetalScratch();
    if (scratch != nullptr) {
      scratch->reset();
    }
    rund::AccelCheck prepare{};
    try {
      prepare =
          resources.program != nullptr &&
                  prepared_step.step->kind() == rund::kernel::NodeKind::Map
              ? PrepareMetalMapRouteStep(
                    pick, prepared_step,
                    std::static_pointer_cast<const MetalMapTemplateResources>(
                        resources.program->steps[index].immutable),
                    entry->resource)
              : PrepareMetalStep(
                    pick, prepared_step, entry->ops,
                    resources.program == nullptr
                        ? nullptr
                        : static_cast<const MetalKernelImmutablePipelines *>(
                              resources.program->steps[index].immutable.get()),
                    entry->resource);
    } catch (const std::bad_alloc &) {
      prepare = rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!prepare.ok) {
      RecordNode(failed_node, steps[index]);
      return prepare;
    }
    if (publish_required &&
        prepared_step.step->kind() != rund::kernel::NodeKind::Map) {
      std::shared_ptr<const MetalKernelImmutablePipelines> frozen;
      const rund::AccelCheck freeze = FreezeMetalPrimitivePipelines(
          prepared_step.step->kind(), entry->resource, frozen);
      if (!freeze.ok) {
        RecordNode(failed_node, steps[index]);
        return freeze;
      }
      resources.program->steps[index].immutable = std::move(frozen);
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
  if (publish_required) {
    if (template_probe == nullptr) {
      return {false, "accel_kernel_template_invalid"};
    }
    const rund::AccelCheck published =
        PublishMetalProgramTemplate(*template_probe, templates, resources);
    if (!published.ok) {
      return published;
    }
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
