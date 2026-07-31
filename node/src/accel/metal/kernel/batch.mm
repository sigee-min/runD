#include "batch/copy.hpp"
#include "batch/encode.hpp"
#include "batch/plan.hpp"

#include "../../kernel/backend/execute.hpp"
#include "../../kernel/reset/stats.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <span>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

void reject(const std::span<rund::AccelCheck> results,
            const rund::AccelCheck failure) noexcept {
  std::fill(results.begin(), results.end(), failure);
}

void finish_packed(const std::span<const BackendBatchEntry> entries,
                   const std::span<rund::AccelCheck> results,
                   const BatchMapGroup &group) {
  for (std::size_t index = group.begin; index < group.begin + group.count;
       ++index) {
    results[index] = rund::AccelCheck{true, "ok"};
    if (entries[index].stats != nullptr) {
      SetResetStats(*entries[index].stats, true, 0u, 0u);
    }
  }
}

void finish_general(MetalAdapter &adapter,
                    const std::span<const BackendBatchEntry> entries,
                    const std::span<rund::AccelCheck> results,
                    const BatchMapGroup &group, rund::RuntimeStats &stats,
                    std::uint64_t &dispatches, rund::AccelCheck &batch) {
  for (std::size_t index = group.begin; index < group.begin + group.count;
       ++index) {
    auto *const resources =
        static_cast<MetalKernelResources *>(entries[index].prepared->get());
    dispatches = ::rund::detail::counter::SaturatingAdd(
        dispatches, resources->dispatch_count);
    stats.reset_command_count = ::rund::detail::counter::SaturatingAdd(
        stats.reset_command_count, resources->reset_count);
    stats.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        stats.reset_bytes, resources->reset_bytes);
    results[index] = FinishMetalSteps(adapter, *resources);
    if (entries[index].stats != nullptr) {
      SetResetStats(*entries[index].stats, results[index].ok,
                    resources->reset_count, resources->reset_bytes);
    }
    if (batch.ok && !results[index].ok) {
      batch = results[index];
    }
  }
}

} // namespace

rund::AccelCheck
RunPreparedMetalBatch(const std::span<const BackendBatchEntry> entries,
                      const std::span<rund::AccelCheck> results,
                      std::shared_ptr<void> &owner, rund::RuntimeStats &stats) {
  stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  @autoreleasepool {
    if (entries.empty() || entries.size() != results.size() ||
        entries.front().run == nullptr ||
        entries.front().run->pick == nullptr) {
      const rund::AccelCheck failure{false, "accel_kernel_run_invalid"};
      reject(results, failure);
      return failure;
    }
    MetalKernelContext context{};
    const rund::AccelCheck valid =
        ValidateMetalKernelContext(*entries.front().run->pick, context);
    if (!valid.ok || context.adapter == nullptr) {
      reject(results, valid);
      return valid;
    }

    metalbatch::Workspace *workspace = nullptr;
    metalbatch::Maps maps{};
    const rund::AccelCheck planned =
        metalbatch::prepare(*context.adapter, entries, owner, workspace, maps);
    if (!planned.ok || workspace == nullptr) {
      const rund::AccelCheck failure =
          planned.ok
              ? rund::AccelCheck{false, "compute_batch_workspace_invalid"}
              : planned;
      reject(results, failure);
      return failure;
    }
    const std::span<const BatchMapView> views{workspace->views.data(),
                                              entries.size()};
    const BatchMapPlan &plan = workspace->plan;
    if (!metalbatch::pack(views, plan, maps, *workspace)) {
      const rund::AccelCheck failure{false, "compute_batch_workspace_invalid"};
      reject(results, failure);
      return failure;
    }

    CommandRun command{};
    const rund::AccelCheck ready =
        OpenCommand<ResourceRefs::Borrowed>(*context.adapter, command);
    if (!ready.ok) {
      reject(results, ready);
      return ready;
    }
    const rund::AccelCheck encoded = metalbatch::encode(
        *context.adapter, entries, views, plan, maps, workspace, command);
    if (!encoded.ok) {
      reject(results, encoded);
      return encoded;
    }
    const rund::AccelCheck submitted =
        WaitCommand(*context.adapter, (__bridge void *)command.buffer, &stats);
    if (!submitted.ok) {
      reject(results, submitted);
      return submitted;
    }
    if (!metalbatch::unpack(views, plan, maps, *workspace)) {
      const rund::AccelCheck failure{false, "compute_batch_workspace_invalid"};
      reject(results, failure);
      return failure;
    }

    rund::AccelCheck batch{true, "ok"};
    std::uint64_t dispatches = 0u;
    for (std::size_t group_index = 0u; group_index < plan.size; ++group_index) {
      const BatchMapGroup &group = plan.groups[group_index];
      if (group.packed) {
        RecordMetalDispatches(*context.adapter, 1u);
        dispatches = ::rund::detail::counter::SaturatingAdd(dispatches, 1u);
        finish_packed(entries, results, group);
      } else {
        finish_general(*context.adapter, entries, results, group, stats,
                       dispatches, batch);
      }
    }
    stats.dispatch_count = batch.ok ? dispatches : 0u;
    if (!batch.ok) {
      stats.reset_command_count = 0u;
      stats.reset_bytes = 0u;
    }
    return batch;
  }
}

#else
rund::AccelCheck
RunPreparedMetalBatch(const std::span<const BackendBatchEntry>,
                      const std::span<rund::AccelCheck> results,
                      std::shared_ptr<void> &, rund::RuntimeStats &stats) {
  stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  const rund::AccelCheck failure{false, "accel_metal_unavailable"};
  std::fill(results.begin(), results.end(), failure);
  return failure;
}
#endif

} // namespace rund::node::accel::detail
