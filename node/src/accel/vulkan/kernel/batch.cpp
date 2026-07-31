#include <accel/check.hpp>

#include "../../kernel/backend/execute.hpp"
#include "../../kernel/reset/stats.hpp"
#include "../command.hpp"
#include "../runtime/timestamp.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void RejectVulkanBatch(const std::span<rund::AccelCheck> results,
                       const rund::AccelCheck failure) noexcept {
  std::fill(results.begin(), results.end(), failure);
}

[[nodiscard]] rund::AccelCheck
ValidateVulkanBatch(const std::span<const BackendBatchEntry> entries,
                    VulkanAdapter *&adapter) {
  adapter = nullptr;
  for (const BackendBatchEntry &entry : entries) {
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<VulkanKernelResources *>(entry.prepared->get());
    if (entry.run == nullptr || entry.run->pick == nullptr ||
        resources == nullptr || resources->size() == 0u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        ValidateVulkanKernelContext(*entry.run->pick, context);
    if (!valid.ok || context.adapter == nullptr) {
      return valid;
    }
    if (adapter == nullptr) {
      adapter = context.adapter;
    } else if (adapter != context.adapter) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  return adapter == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
EncodeVulkanBatch(VulkanAdapter &adapter,
                  const std::span<const BackendBatchEntry> entries) {
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  for (const BackendBatchEntry &entry : entries) {
    auto &resources =
        *static_cast<VulkanKernelResources *>(entry.prepared->get());
    const rund::AccelCheck executed = ExecuteVulkanKernel(adapter, resources);
    if (!executed.ok) {
      return executed;
    }
  }
  EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck
RunPreparedVulkanBatch(const std::span<const BackendBatchEntry> entries,
                       const std::span<rund::AccelCheck> results,
                       std::shared_ptr<void> &workspace,
                       rund::RuntimeStats &stats) {
  stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  (void)workspace;
  if (entries.empty() || entries.size() != results.size()) {
    const rund::AccelCheck failure{false, "accel_kernel_run_invalid"};
    RejectVulkanBatch(results, failure);
    return failure;
  }
  VulkanAdapter *adapter = nullptr;
  const rund::AccelCheck valid = ValidateVulkanBatch(entries, adapter);
  if (!valid.ok) {
    RejectVulkanBatch(results, valid);
    return valid;
  }

  std::lock_guard lock{adapter->mutex};
  if (!EnsureVulkanCommandResources(*adapter) ||
      !BeginVulkanCommand(*adapter)) {
    const rund::AccelCheck failure{false, VulkanLastError(adapter)};
    RejectVulkanBatch(results, failure);
    return failure;
  }
  const rund::AccelCheck encoded = EncodeVulkanBatch(*adapter, entries);
  if (!encoded.ok) {
    CancelVulkanCommand(*adapter);
    RejectVulkanBatch(results, encoded);
    return encoded;
  }
  if (!SubmitVulkanCommand(*adapter, true, &stats)) {
    const rund::AccelCheck failure{false, VulkanLastError(adapter)};
    RejectVulkanBatch(results, failure);
    return failure;
  }

  rund::AccelCheck batch{true, "ok"};
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    auto *const resources =
        static_cast<VulkanKernelResources *>(entries[index].prepared->get());
    stats.dispatch_count = ::rund::detail::counter::SaturatingAdd(
        stats.dispatch_count, resources->dispatch_count);
    stats.reset_command_count = ::rund::detail::counter::SaturatingAdd(
        stats.reset_command_count, resources->reset_count);
    stats.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        stats.reset_bytes, resources->reset_bytes);
    results[index] = FinishVulkanSteps(*adapter, *resources);
    if (entries[index].stats != nullptr) {
      SetResetStats(*entries[index].stats, results[index].ok,
                    resources->reset_count, resources->reset_bytes);
    }
    if (batch.ok && !results[index].ok) {
      batch = results[index];
    }
  }
  if (!batch.ok) {
    stats.dispatch_count = 0u;
    stats.reset_command_count = 0u;
    stats.reset_bytes = 0u;
  }
  return batch;
}

#else
rund::AccelCheck
RunPreparedVulkanBatch(const std::span<const BackendBatchEntry>,
                       const std::span<rund::AccelCheck> results,
                       std::shared_ptr<void> &, rund::RuntimeStats &stats) {
  stats = rund::RuntimeStats{.ok = true, .reason = "ok"};
  const rund::AccelCheck failure{false, "accel_vulkan_loader_unavailable"};
  std::fill(results.begin(), results.end(), failure);
  return failure;
}
#endif

} // namespace rund::node::accel::detail
