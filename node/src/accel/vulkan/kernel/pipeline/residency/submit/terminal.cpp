#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../runtime/counter.hpp"
#include "../../../../timeline/owner.hpp"
#include "../../../control.hpp"

#include <rund/compute/reason.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace vulkan_residency_submit_detail {
namespace {

constexpr std::uint64_t WindowTimeoutNs = 30'000'000'000u;

[[nodiscard]] bool OpenSuppressed(VulkanResidencyWindowRun &run,
                                  const std::size_t index,
                                  const char *const reason) noexcept {
  if (index >= run.request.batch_count || reason == nullptr ||
      reason[0] == '\0') {
    return false;
  }
  const BackendResidencyWindowBatch &source = run.request.batches[index];
  VulkanResidencyWindowBatch &batch = run.batches[index];
  VulkanPipeline *const pipeline = batch.pipeline;
  VulkanResidencySelection *const selection =
      pipeline == nullptr ? nullptr : pipeline->residency.get();
  const BackendResidencyWindowSignal signal{
      .admission = rund::AccelCheck{false, reason},
      .plan_identity = run.request.plan_identity,
      .token = run.request.token,
      .generation = run.request.generation,
      .epoch = source.epoch,
      .control_generation = source.control_generation,
      .bank = source.bank,
  };
  if (pipeline == nullptr || selection == nullptr || batch.signaled ||
      !CanSignalVulkanTimelineReady(pipeline->adapter->timeline, batch.point)
           .ok ||
      !SelectVulkanResidencyArguments(
          *pipeline,
          std::span<const std::uint32_t>{source.locals.data(),
                                         source.local_count},
          false) ||
      !SeedVulkanResidencyControl(*pipeline, signal, false) ||
      !SignalVulkanTimelineReady(pipeline->adapter->timeline, batch.point).ok) {
    return false;
  }
  batch.admission = signal.admission;
  batch.signaled = true;
  ++run.signaled_count;
  run.inflight_peak =
      std::max(run.inflight_peak, run.signaled_count - run.release_count);
  return true;
}

void UnknownFinal(VulkanResidencyWindowRun &run,
                  const std::size_t first) noexcept {
  BackendResidencyWindowRequest callbacks{};
  std::array<BackendResidencyWindowRelease, VulkanResidencyWindowCapacity>
      releases{};
  BackendResidencyWindowFinal final{};
  {
    std::lock_guard lock{run.gate};
    if (!run.active || run.final_sent) {
      return;
    }
    callbacks = run.request;
    for (std::size_t index = first; index < run.request.batch_count; ++index) {
      const BackendResidencyWindowBatch &source = run.request.batches[index];
      const KernelResult result{
          .check = rund::AccelCheck{false, "compute_device_lost"},
          .stats =
              rund::RuntimeStats{
                  .outcome = {.ok = false, .reason = "compute_device_lost"}},
          .pipeline = {.submitted = true},
          .terminal = NativeTerminal::UnknownMayWrite,
      };
      run.receipts[index] = BackendResidencyWindowReceipt{
          .check = result.check,
          .terminal = NativeTerminal::UnknownMayWrite,
          .epoch = source.epoch,
          .backend_sequence = source.epoch + 1u,
          .bank = source.bank,
          .dispatched = true,
          .completed = false,
          .may_write = true,
      };
      releases[index - first] = BackendResidencyWindowRelease{
          .receipt = run.receipts[index], .result = result};
    }
    run.release_count = run.request.batch_count;
    run.final_sent = true;
    run.quarantined = true;
    run.batches[0u].pipeline->adapter->residency_quarantined.store(
        true, std::memory_order_release);
    final = BackendResidencyWindowFinal{
        .check = rund::AccelCheck{false, "compute_device_lost"},
        .terminal = NativeTerminal::UnknownMayWrite,
        .plan_identity = run.request.plan_identity,
        .token = run.request.token,
        .generation = run.request.generation,
        .first_epoch = run.request.first_epoch,
        .public_handoffs = 1u,
        .native_batches = run.request.batch_count,
        .queue_calls = 1u,
        .native_inflight_peak = std::max<std::size_t>(1u, run.inflight_peak),
        .receipts = run.receipts,
        .receipt_count = run.request.batch_count,
        .completed_ns = MonotonicNanoseconds(),
    };
  }
  for (std::size_t index = first; index < callbacks.batch_count; ++index) {
    callbacks.release(callbacks.user, std::move(releases[index - first]));
  }
  callbacks.final(callbacks.user, std::move(final));
  // Unknown retains native prepared owners as a quarantine, never the raw
  // common completion control that may die as soon as Final returns.
  {
    std::lock_guard lock{run.gate};
    run.request.release = nullptr;
    run.request.final = nullptr;
    run.request.user = nullptr;
  }
}

} // namespace

void ServiceVulkanResidencyWindow(void *const raw) noexcept {
  auto *const run = static_cast<VulkanResidencyWindowRun *>(raw);
  if (run == nullptr) {
    return;
  }
  for (std::size_t index = 0u;; ++index) {
    VulkanTimelinePoint point{};
    VulkanAdapter *adapter = nullptr;
    {
      std::unique_lock lock{run->gate};
      if (!run->active || run->final_sent ||
          index >= run->request.batch_count) {
        return;
      }
      const bool ready = run->ready.wait_for(
          lock, std::chrono::nanoseconds{WindowTimeoutNs}, [&] {
            return !run->active || run->final_sent || run->aborting ||
                   run->batches[index].signaled;
          });
      if (!run->active || run->final_sent) {
        return;
      }
      const char *const suppression =
          run->aborting ? run->abort_failure.reason : "compute_backend_failed";
      if ((!ready || run->aborting) && !run->batches[index].signaled &&
          !OpenSuppressed(*run, index, suppression)) {
        lock.unlock();
        UnknownFinal(*run, index);
        return;
      }
      point = run->batches[index].point;
      adapter = run->batches[index].pipeline->adapter;
    }
    const rund::AccelCheck waited =
        WaitVulkanTimelineDone(adapter->timeline, point, WindowTimeoutNs);
    if (!waited.ok) {
      UnknownFinal(*run, index);
      return;
    }

    BackendResidencyWindowRelease release{};
    BackendResidencyWindowRequest callbacks{};
    bool last = false;
    {
      std::lock_guard lock{run->gate};
      const BackendResidencyWindowBatch &source = run->request.batches[index];
      const bool aborted = run->aborting;
      KernelResult result =
          aborted
              ? KernelResult{
                    .check = rund::AccelCheck{false, "compute_device_lost"},
                    .stats = rund::RuntimeStats{
                        .outcome = {.ok = false,
                                    .reason = "compute_device_lost"}},
                    .pipeline = {.submitted = true},
                    .terminal = NativeTerminal::UnknownMayWrite,
                }
              : ObserveVulkanResidency(
                    *run->batches[index].pipeline,
                    run->batches[index].admission,
                    run->batches[index].dispatch_count,
                    run->batches[index].control_count,
                    run->batches[index].reset_count,
                    run->batches[index].reset_bytes,
                    source.control_generation);
      run->receipts[index] = BackendResidencyWindowReceipt{
          .check = result.check,
          .terminal =
              aborted ? NativeTerminal::UnknownMayWrite : NativeTerminal::Known,
          .epoch = source.epoch,
          .backend_sequence = source.epoch + 1u,
          .bank = source.bank,
          .dispatched = true,
          .completed = !aborted,
          .may_write = aborted || run->batches[index].admission.ok,
      };
      release = BackendResidencyWindowRelease{.receipt = run->receipts[index],
                                              .result = std::move(result)};
      ++run->release_count;
      callbacks = run->request;
      last = run->release_count == run->request.batch_count;
    }
    callbacks.release(callbacks.user, std::move(release));
    if (!last) {
      continue;
    }

    const rund::AccelCheck closed =
        CloseVulkanTimelineGeneration(adapter->timeline, point);
    BackendResidencyWindowFinal final{};
    {
      std::lock_guard lock{run->gate};
      const BackendResidencyWindowReceipt *failure = nullptr;
      for (std::size_t current = 0u; current < run->request.batch_count;
           ++current) {
        if (failure == nullptr && !run->receipts[current].check.ok) {
          failure = &run->receipts[current];
        }
        auto *const pipeline = static_cast<VulkanPipeline *>(
            run->request.batches[current].prepared.get());
        if (pipeline != nullptr && pipeline->residency != nullptr) {
          VulkanResidencyWindowRun *expected = run;
          static_cast<void>(
              pipeline->residency->active_window.compare_exchange_strong(
                  expected, nullptr, std::memory_order_acq_rel,
                  std::memory_order_acquire));
        }
      }
      final = BackendResidencyWindowFinal{
          .check = !closed.ok
                       ? closed
                       : (failure == nullptr ? rund::AccelCheck{true, "ok"}
                                             : failure->check),
          .terminal = run->aborting ? NativeTerminal::UnknownMayWrite
                                    : NativeTerminal::Known,
          .plan_identity = run->request.plan_identity,
          .token = run->request.token,
          .generation = run->request.generation,
          .first_epoch = run->request.first_epoch,
          .public_handoffs = 1u,
          .native_batches = run->request.batch_count,
          .queue_calls = 1u,
          .native_inflight_peak = run->inflight_peak,
          .receipts = run->receipts,
          .receipt_count = run->request.batch_count,
          .completed_ns = MonotonicNanoseconds(),
      };
      callbacks = run->request;
      run->final_sent = true;
      if (!run->aborting) {
        run->request = {};
        run->batches = {};
        run->receipts = {};
        run->release_count = 0u;
        run->signaled_count = 0u;
        run->inflight_peak = 0u;
        run->timeline_generation = 0u;
        run->active = false;
      } else {
        run->request.release = nullptr;
        run->request.final = nullptr;
        run->request.user = nullptr;
      }
    }
    callbacks.final(callbacks.user, std::move(final));
    return;
  }
}

} // namespace vulkan_residency_submit_detail

#endif

} // namespace rund::node::accel::detail
