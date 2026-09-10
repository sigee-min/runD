#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../timeline/owner.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace rund::node::accel::detail::schedule {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool OpenSuppressed(VulkanResidencyScheduleRun &run,
                                  const std::uint64_t epoch,
                                  const char *const reason) noexcept {
  if (reason == nullptr || reason[0u] == '\0' ||
      epoch >= run.request.epoch_count) {
    return false;
  }
  const std::size_t role =
      static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity);
  VulkanResidencyScheduleCell &cell = run.cells[role];
  const BackendResidencyScheduleRole &source = SourceRole(run, epoch);
  VulkanPipeline *const pipeline = NativeRole(run, epoch).pipeline;
  std::uint32_t generation = 0u;
  const BackendResidencyWindowSignal signal{
      .admission = {false, reason},
      .plan_identity = run.request.plan_identity,
      .token = run.request.token,
      .generation = run.request.generation,
      .epoch = epoch,
      .control_generation =
          ControlGeneration(source, epoch, generation) ? generation : 0u,
      .bank = source.bank,
  };
  const std::size_t count = LocalCount(run, epoch);
  const VulkanTimelinePoint ready = Point(run, epoch);
  if (pipeline == nullptr || cell.signaled ||
      signal.control_generation == 0u ||
      !CanSignalVulkanTimelineReady(pipeline->adapter->timeline, ready).ok ||
      !SelectVulkanResidencyArguments(
          *pipeline, std::span<const std::uint32_t>{source.locals.data(), count},
          false) ||
      !SeedVulkanResidencyControl(*pipeline, signal, false) ||
      !SignalVulkanTimelineReady(pipeline->adapter->timeline, ready).ok) {
    return false;
  }
  cell.admission = signal.admission;
  cell.epoch = epoch;
  cell.signaled = true;
  ++run.signaled_count;
  run.inflight_peak =
      std::max(run.inflight_peak, run.signaled_count - run.release_count);
  return true;
}

void UnknownFinal(VulkanResidencyScheduleRun &run,
                  const std::uint64_t epoch) noexcept {
  BackendResidencyScheduleRequest callbacks{};
  BackendResidencyScheduleRelease release{};
  BackendResidencyScheduleFinal final{};
  {
    std::lock_guard lock{run.gate};
    if (!run.active || run.final_sent) {
      return;
    }
    const BackendResidencyScheduleRole &source = SourceRole(run, epoch);
    const KernelResult result{
        .check = {false, "compute_device_lost"},
        .stats =
            rund::RuntimeStats{
                .run = {.work = {.command_submit_count = 1u,
                                 .command_inflight_peak =
                                     std::max<std::uint64_t>(
                                         1u, run.inflight_peak)}},
                .outcome = {.ok = false, .reason = "compute_device_lost"}},
        .pipeline = {.submitted = true},
        .terminal = NativeTerminal::UnknownMayWrite,
    };
    release = BackendResidencyScheduleRelease{
        .receipt =
            BackendResidencyWindowReceipt{
                .check = result.check,
                .terminal = NativeTerminal::UnknownMayWrite,
                .epoch = epoch,
                .backend_sequence = epoch + 1u,
                .bank = source.bank,
                .dispatched = true,
                .completed = false,
                .may_write = true,
            },
        .result = result,
    };
    callbacks = run.request;
    run.first_failure = result.check;
    run.release_count = epoch + 1u;
    run.final_sent = true;
    run.quarantined = true;
    NativeRole(run, epoch).pipeline->adapter->residency_quarantined.store(
        true, std::memory_order_release);
    final = BackendResidencyScheduleFinal{
        .check = result.check,
        .terminal = NativeTerminal::UnknownMayWrite,
        .plan_identity = run.request.plan_identity,
        .token = run.request.token,
        .generation = run.request.generation,
        .epoch_count = run.request.epoch_count,
        .public_handoffs = 1u,
        .native_batches = run.request.epoch_count,
        .queue_calls = 1u,
        .native_inflight_peak = std::max<std::uint64_t>(1u, run.inflight_peak),
        .released_prefix = run.release_count,
        .completed_prefix = run.success_prefix,
        .completed_ns = MonotonicNanoseconds(),
    };
    run.request.release = nullptr;
    run.request.final = nullptr;
    run.request.user = nullptr;
  }
  callbacks.release(callbacks.user, std::move(release));
  callbacks.final(callbacks.user, std::move(final));
}

} // namespace

void Service(void *const raw) noexcept {
  auto *const run = static_cast<VulkanResidencyScheduleRun *>(raw);
  if (run == nullptr) {
    return;
  }
  for (std::uint64_t epoch = 0u;; ++epoch) {
    VulkanTimelinePoint terminal{};
    VulkanAdapter *adapter = nullptr;
    {
      std::unique_lock lock{run->gate};
      if (!run->active || run->final_sent ||
          epoch >= run->request.epoch_count) {
        return;
      }
      const std::size_t role =
          static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity);
      VulkanResidencyScheduleCell &cell = run->cells[role];
      const bool ready = run->ready.wait_for(
          lock, std::chrono::nanoseconds{ScheduleTimeoutNs}, [&] {
            return !run->active || run->final_sent || run->aborting ||
                   (cell.signaled && cell.epoch == epoch);
          });
      if (!run->active || run->final_sent) {
        return;
      }
      const char *const reason =
          run->aborting ? run->abort_failure.reason : "compute_backend_failed";
      if ((!ready || run->aborting) &&
          !(cell.signaled && cell.epoch == epoch) &&
          !OpenSuppressed(*run, epoch, reason)) {
        lock.unlock();
        UnknownFinal(*run, epoch);
        return;
      }
      terminal = Point(*run, epoch);
      adapter = NativeRole(*run, epoch).pipeline->adapter;
    }
    const rund::AccelCheck waited =
        WaitVulkanTimelineDone(adapter->timeline, terminal, ScheduleTimeoutNs);
    if (!waited.ok) {
      UnknownFinal(*run, epoch);
      return;
    }

    BackendResidencyScheduleRelease release{};
    BackendResidencyScheduleRequest callbacks{};
    bool last = false;
    std::uint32_t generation = 0u;
    if (!ControlGeneration(SourceRole(*run, epoch), epoch, generation)) {
      UnknownFinal(*run, epoch);
      return;
    }
    {
      std::lock_guard lock{run->gate};
      const std::size_t role =
          static_cast<std::size_t>(epoch % ResidencyScheduleRoleCapacity);
      VulkanResidencyScheduleCell &cell = run->cells[role];
      const BackendResidencyScheduleRole &source = SourceRole(*run, epoch);
      VulkanResidencyScheduleRole &native = NativeRole(*run, epoch);
      KernelResult result =
          run->aborting
              ? KernelResult{
                    .check = {false, "compute_device_lost"},
                    .stats = rund::RuntimeStats{
                        .outcome = {.ok = false,
                                    .reason = "compute_device_lost"}},
                    .pipeline = {.submitted = true},
                    .terminal = NativeTerminal::UnknownMayWrite,
                }
              : ObserveVulkanResidency(
                    *native.pipeline, cell.admission, native.dispatch_count,
                    native.control_count, native.reset_count,
                    native.reset_bytes, generation);
      const bool unknown = result.terminal == NativeTerminal::UnknownMayWrite ||
                           run->aborting;
      release = BackendResidencyScheduleRelease{
          .receipt =
              BackendResidencyWindowReceipt{
                  .check = result.check,
                  .terminal = unknown ? NativeTerminal::UnknownMayWrite
                                      : NativeTerminal::Known,
                  .epoch = epoch,
                  .backend_sequence = epoch + 1u,
                  .bank = source.bank,
                  .dispatched = true,
                  .completed = !unknown,
                  .may_write = unknown || cell.admission.ok,
              },
          .result = std::move(result),
      };
      if (release.receipt.check.ok && run->first_failure.ok) {
        ++run->success_prefix;
      } else if (!release.receipt.check.ok && run->first_failure.ok) {
        run->first_failure = release.receipt.check;
      } else if (!release.receipt.may_write && release.receipt.completed) {
        if (run->suppressed_count == 0u) {
          run->suppressed_first = epoch;
        }
        ++run->suppressed_count;
      }
      cell = {};
      ++run->release_count;
      callbacks = run->request;
      last = run->release_count == run->request.epoch_count;
    }
    callbacks.release(callbacks.user, std::move(release));
    if (!last) {
      continue;
    }

    const rund::AccelCheck closed =
        CloseVulkanTimelineGeneration(adapter->timeline, terminal);
    BackendResidencyScheduleFinal final{};
    {
      std::lock_guard lock{run->gate};
      final = BackendResidencyScheduleFinal{
          .check = !closed.ok ? closed : run->first_failure,
          .terminal = run->aborting ? NativeTerminal::UnknownMayWrite
                                    : NativeTerminal::Known,
          .plan_identity = run->request.plan_identity,
          .token = run->request.token,
          .generation = run->request.generation,
          .epoch_count = run->request.epoch_count,
          .public_handoffs = 1u,
          .native_batches = run->request.epoch_count,
          .queue_calls = 1u,
          .native_inflight_peak = run->inflight_peak,
          .released_prefix = run->release_count,
          .completed_prefix = run->success_prefix,
          .suppressed_first = run->suppressed_first,
          .suppressed_count = run->suppressed_count,
          .completed_ns = MonotonicNanoseconds(),
      };
      callbacks = run->request;
      run->final_sent = true;
      ClearActive(*run);
      if (!run->aborting) {
        run->request = {};
        run->roles = {};
        run->tail = {};
        run->cells = {};
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

#endif

} // namespace rund::node::accel::detail::schedule
