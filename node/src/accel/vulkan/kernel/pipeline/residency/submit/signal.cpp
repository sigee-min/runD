#include "internal.hpp"

#include "../../../../timeline/owner.hpp"

#include <algorithm>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck SignalVulkanResidencyWindow(
    const std::shared_ptr<void> &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
  using vulkan_residency_submit_detail::Invalid;
  using vulkan_residency_submit_detail::Unavailable;
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  VulkanResidencySelection *const selection =
      pipeline == nullptr ? nullptr : pipeline->residency.get();
  VulkanResidencyWindowRun *const run =
      selection == nullptr
          ? nullptr
          : selection->active_window.load(std::memory_order_acquire);
  if (!ValidVulkanPipeline(pipeline) || selection == nullptr ||
      run == nullptr || signal.plan_identity == 0u || signal.token == 0u ||
      signal.generation == 0u || signal.control_generation == 0u) {
    return Invalid();
  }
  std::lock_guard lock{run->gate};
  if (!run->active || run->final_sent || run->quarantined ||
      signal.plan_identity != run->request.plan_identity ||
      signal.token != run->request.token ||
      signal.generation != run->request.generation ||
      signal.epoch < run->request.first_epoch ||
      signal.epoch - run->request.first_epoch >= run->request.batch_count) {
    return Invalid();
  }
  const std::size_t index =
      static_cast<std::size_t>(signal.epoch - run->request.first_epoch);
  const BackendResidencyWindowBatch &source = run->request.batches[index];
  VulkanResidencyWindowBatch &batch = run->batches[index];
  if (source.prepared.get() != pipeline || source.bank != signal.bank ||
      source.control_generation != signal.control_generation ||
      batch.pipeline != pipeline || batch.signaled ||
      (index >= 2u && !run->receipts[index - 2u].completed) ||
      signal.admission.reason == nullptr ||
      signal.admission.reason[0u] == '\0') {
    return Invalid();
  }
  const rund::AccelCheck admitted =
      CanSignalVulkanTimelineReady(pipeline->adapter->timeline, batch.point);
  if (!admitted.ok) {
    return admitted;
  }
  if (!SelectVulkanResidencyArguments(
          *pipeline,
          std::span<const std::uint32_t>{source.locals.data(),
                                         source.local_count},
          signal.admission.ok) ||
      !SeedVulkanResidencyControl(*pipeline, signal, signal.admission.ok)) {
    return Unavailable();
  }
  const rund::AccelCheck signaled =
      SignalVulkanTimelineReady(pipeline->adapter->timeline, batch.point);
  if (!signaled.ok) {
    return signaled;
  }
  batch.admission = signal.admission;
  batch.signaled = true;
  ++run->signaled_count;
  run->inflight_peak =
      std::max(run->inflight_peak, run->signaled_count - run->release_count);
  run->ready.notify_one();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
