#include "../../adapter/error.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../kernel/reset/stats.hpp"
#include "../../runtime/timestamp.hpp"
#include "../local.hpp"
#include "../trace.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
BeginVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources,
                 const bool collect_timestamp) {
  if (!EnsureVulkanCommandResources(adapter) || !BeginVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (collect_timestamp) {
    BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  }
  const rund::AccelCheck encoded = ExecuteVulkanKernel(adapter, resources);
  if (!encoded.ok) {
    CancelVulkanCommand(adapter);
  } else if (collect_timestamp) {
    EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  }
  return encoded;
}

[[nodiscard]] rund::AccelCheck
RunVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources) {
  const rund::AccelCheck encoded = BeginVulkanSteps(adapter, resources, true);
  if (!encoded.ok) {
    return encoded;
  }
  if (!SubmitVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}

void CompleteVulkanPrepared(void *const raw, KernelResult submitted) noexcept {
  auto *const state =
      static_cast<submission::State<VulkanKernelResources> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<VulkanKernelResources> claim =
      submission::Take(*state);
  if (!claim) {
    return;
  }
  VulkanKernelResources &resources = *claim.owner;
  const bool trace_active = resources.trace_active;
  resources.trace_active = false;
  if (submitted.check.ok && resources.adapter != nullptr) {
    std::lock_guard lock{resources.adapter->mutex};
    if (trace_active) {
      submitted.check = FoldVulkanDispatchTrace(
          *resources.adapter, resources.trace, submitted.stats);
    }
    if (submitted.check.ok) {
      submitted.check =
          FinishVulkanSteps(*resources.adapter, resources, &submitted.stats);
    }
  }
  submitted.stats.run.work.dispatch_count =
      submitted.check.ok ? resources.dispatch_count : 0u;
  SetResetStats(submitted.stats, submitted.check.ok, resources.reset_count,
                resources.reset_bytes);
  claim.completion(claim.user, submitted);
}

} // namespace
#endif

std::uint64_t
VulkanKernelTraffic(const std::shared_ptr<void> &prepared) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const resources =
      static_cast<const VulkanKernelResources *>(prepared.get());
  return resources == nullptr ? 0u : resources->traffic;
#else
  (void)prepared;
  return 0u;
#endif
}

rund::AccelCheck RunVulkanResources(const rund::AccelDevice &pick,
                                    const std::shared_ptr<void> &prepared) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const resources = static_cast<VulkanKernelResources *>(prepared.get());
  if (resources == nullptr || resources->size() == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  VulkanKernelContext context{};
  const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
  if (!valid.ok) {
    return valid;
  }
  VulkanAdapter *const adapter = context.adapter;
  std::lock_guard<std::mutex> lock{adapter->mutex};
  const rund::AccelCheck executed = RunVulkanSteps(*adapter, *resources);
  if (!executed.ok) {
    return executed;
  }
  return FinishVulkanSteps(*adapter, *resources);
#else
  (void)pick;
  (void)prepared;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck SubmitVulkanResources(const rund::AccelDevice &pick,
                                       const std::shared_ptr<void> &prepared,
                                       const KernelCompletion completion,
                                       void *const user,
                                       PreparedMemoryMeter *const memory,
                                       const KernelTiming timing) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const resources = static_cast<VulkanKernelResources *>(prepared.get());
  if (resources == nullptr || resources->size() == 0u ||
      completion == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  VulkanKernelContext context{};
  const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
  if (!valid.ok) {
    return valid;
  }
  submission::State<VulkanKernelResources> &state = resources->submission;
  if (!submission::Begin(state, *resources, completion, user)) {
    return rund::AccelCheck{false, "compute_job_busy"};
  }
  VulkanAdapter *const adapter = context.adapter;
  rund::AccelCheck submitted{};
  {
    std::lock_guard lock{adapter->mutex};
    if (timing == KernelTiming::Dispatch) {
      submitted = EnsureVulkanDispatchTrace(*adapter, *resources, memory);
      if (!submitted.ok) {
        submission::Cancel(state);
        return submitted;
      }
    }
    const bool collect_timestamp = timing == KernelTiming::Submission;
    if (!EnsureVulkanCommandResources(*adapter) ||
        !BeginVulkanCommand(*adapter)) {
      submitted = rund::AccelCheck{false, VulkanLastError(adapter)};
    } else if (timing == KernelTiming::Dispatch) {
      submitted = EncodeVulkanDispatchTrace(*adapter, *resources);
      if (!submitted.ok) {
        CancelVulkanCommand(*adapter);
      }
    } else {
      if (collect_timestamp) {
        BeginVulkanTimestampSpan(*adapter, adapter->command_buffer);
      }
      submitted = ExecuteVulkanKernel(*adapter, *resources);
      if (!submitted.ok) {
        CancelVulkanCommand(*adapter);
      } else if (collect_timestamp) {
        EndVulkanTimestampSpan(*adapter, adapter->command_buffer);
      }
    }
    if (submitted.ok && !SubmitVulkanCommand(*adapter, CompleteVulkanPrepared,
                                             &state, collect_timestamp)) {
      submitted = rund::AccelCheck{false, VulkanLastError(adapter)};
    }
    resources->trace_active = submitted.ok && timing == KernelTiming::Dispatch;
  }
  if (!submitted.ok) {
    submission::Cancel(state);
    return submitted;
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)prepared;
  (void)completion;
  (void)user;
  (void)memory;
  (void)timing;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
