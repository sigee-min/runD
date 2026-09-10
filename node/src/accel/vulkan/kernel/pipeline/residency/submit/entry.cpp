#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../runtime/counter.hpp"
#include "../../../../timeline/owner.hpp"
#include "../../../control.hpp"

#include <rund/compute/reason.hpp>

#include <array>
#include <limits>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace vulkan_residency_submit_detail {

rund::AccelCheck Invalid() noexcept {
  return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
}

rund::AccelCheck Unavailable() noexcept {
  return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
}

} // namespace vulkan_residency_submit_detail

namespace {

[[nodiscard]] bool
ValidRequest(const BackendResidencyWindowRequest &request) noexcept {
  if (request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.batch_count == 0u ||
      request.batch_count > VulkanResidencyWindowCapacity ||
      request.release == nullptr || request.final == nullptr ||
      request.user == nullptr) {
    return false;
  }
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    const BackendResidencyWindowBatch &batch = request.batches[index];
    if (request.first_epoch >
            std::numeric_limits<std::uint64_t>::max() - index ||
        batch.prepared == nullptr ||
        batch.epoch != request.first_epoch + index ||
        batch.bank != batch.epoch % 2u || batch.control_generation == 0u ||
        batch.local_count == 0u ||
        batch.local_count > ResidencyWindowLocalCapacity) {
      return false;
    }
    for (std::size_t local = 0u; local < batch.local_count; ++local) {
      for (std::size_t prior = 0u; prior < local; ++prior) {
        if (batch.locals[local] == batch.locals[prior]) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool Materialize(
    const BackendResidencyWindowRequest &request,
    std::array<VulkanResidencyWindowBatch, VulkanResidencyWindowCapacity>
        &batches,
    VulkanAdapter *&adapter, VulkanResidencyWindowRun *&run) noexcept {
  adapter = nullptr;
  run = nullptr;
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    const BackendResidencyWindowBatch &source = request.batches[index];
    auto *const pipeline = static_cast<VulkanPipeline *>(source.prepared.get());
    VulkanResidencySelection *const selection =
        pipeline == nullptr ? nullptr : pipeline->residency.get();
    if (!ValidVulkanPipeline(pipeline) || selection == nullptr ||
        !selection->ready || !selection->bounded ||
        selection->adapter != pipeline->adapter ||
        selection->active_window.load(std::memory_order_acquire) != nullptr ||
        (adapter != nullptr && adapter != pipeline->adapter) ||
        (index != 0u &&
         batches[index - 1u].pipeline->residency.get() == selection)) {
      return false;
    }
    adapter = pipeline->adapter;
    VulkanResidencyWindowBatch &target = batches[index];
    target.pipeline = pipeline;
    const rund::AccelCheck built = BuildVulkanResidencySubmission(
        *pipeline,
        std::span<const std::uint32_t>{source.locals.data(),
                                       source.local_count},
        target.commands, target.command_count, target.dispatch_count,
        target.control_count, target.reset_count, target.reset_bytes);
    if (!built.ok || target.command_count == 0u ||
        target.dispatch_count == 0u) {
      return false;
    }
    if (index == 0u) {
      run = &selection->window;
    }
  }
  return adapter != nullptr && run != nullptr;
}

} // namespace

rund::AccelCheck SubmitVulkanResidencyWindow(
    const BackendResidencyWindowRequest &request) noexcept {
  using vulkan_residency_submit_detail::Invalid;
  using vulkan_residency_submit_detail::Unavailable;
  if (!ValidRequest(request)) {
    return Invalid();
  }
  std::array<VulkanResidencyWindowBatch, VulkanResidencyWindowCapacity>
      batches{};
  VulkanAdapter *adapter = nullptr;
  VulkanResidencyWindowRun *run = nullptr;
  if (!Materialize(request, batches, adapter, run) ||
      adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return Invalid();
  }
  std::unique_lock run_lock{run->gate, std::try_to_lock};
  if (!run_lock.owns_lock() || run->active || run->quarantined) {
    return rund::AccelCheck{false, "compute_pipeline_busy"};
  }
  std::unique_lock adapter_lock{adapter->mutex};
  std::unique_lock residency_lock{adapter->residency_mutex};
  if (adapter->residency_stop || adapter->residency_service != nullptr) {
    return Unavailable();
  }

  const auto clear_active = [&] {
    for (std::size_t index = 0u; index < request.batch_count; ++index) {
      VulkanResidencyWindowRun *expected = run;
      static_cast<void>(
          batches[index]
              .pipeline->residency->active_window.compare_exchange_strong(
                  expected, nullptr, std::memory_order_acq_rel,
                  std::memory_order_acquire));
    }
  };
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    VulkanResidencyWindowRun *expected = nullptr;
    if (!batches[index]
             .pipeline->residency->active_window.compare_exchange_strong(
                 expected, run, std::memory_order_acq_rel,
                 std::memory_order_acquire) &&
        expected != run) {
      clear_active();
      return rund::AccelCheck{false, "compute_pipeline_busy"};
    }
  }

  std::uint32_t generation = 0u;
  rund::AccelCheck prepared =
      ReserveVulkanTimelineGeneration(adapter->timeline, generation);
  for (std::size_t index = 0u; prepared.ok && index < request.batch_count;
       ++index) {
    prepared = ReserveVulkanTimelinePoint(adapter->timeline, generation,
                                          batches[index].point);
  }
  if (!prepared.ok) {
    if (generation != 0u) {
      static_cast<void>(
          CancelVulkanTimelineGeneration(adapter->timeline, generation));
    }
    clear_active();
    return prepared;
  }
  std::array<VulkanTimelineBatch, VulkanResidencyWindowCapacity> timeline{};
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    timeline[index] = VulkanTimelineBatch{
        .commands =
            std::span<const VkCommandBuffer>{batches[index].commands.data(),
                                             batches[index].command_count},
        .point = batches[index].point,
    };
  }
  std::uint64_t queue_calls = 0u;
  const rund::AccelCheck submitted =
      SubmitVulkanTimelineWindow(*adapter,
                                 std::span<const VulkanTimelineBatch>{
                                     timeline.data(), request.batch_count},
                                 queue_calls, true);
  if (!submitted.ok || queue_calls != 1u) {
    static_cast<void>(
        CancelVulkanTimelineGeneration(adapter->timeline, generation));
    clear_active();
    return submitted.ok ? Unavailable() : submitted;
  }

  run->request = request;
  run->batches = batches;
  run->receipts = {};
  run->timeline_generation = generation;
  run->release_count = 0u;
  run->signaled_count = 0u;
  run->inflight_peak = 0u;
  run->submit_begin_ns = MonotonicNanoseconds();
  run->active = true;
  run->abort_failure = {};
  run->aborting = false;
  run->final_sent = false;
  adapter->residency_user = run;
  adapter->residency_service =
      vulkan_residency_submit_detail::ServiceVulkanResidencyWindow;
  run_lock.unlock();
  residency_lock.unlock();
  adapter_lock.unlock();
  adapter->residency_cv.notify_one();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
