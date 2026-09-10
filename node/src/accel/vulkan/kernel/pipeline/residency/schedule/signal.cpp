#include "internal.hpp"

#include "../../../../timeline/owner.hpp"

#include <algorithm>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

using namespace schedule;

rund::AccelCheck SignalVulkanResidencySchedule(
    const std::shared_ptr<void> &prepared,
    const BackendResidencyWindowSignal &signal) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  VulkanResidencySelection *const selection =
      pipeline == nullptr ? nullptr : pipeline->residency.get();
  VulkanResidencyScheduleRun *const run =
      selection == nullptr
          ? nullptr
          : selection->active_schedule.load(std::memory_order_acquire);
  if (!ValidVulkanPipeline(pipeline) || run == nullptr ||
      signal.plan_identity == 0u || signal.token == 0u ||
      signal.generation == 0u || signal.control_generation == 0u) {
    return Invalid();
  }
  std::lock_guard lock{run->gate};
  if (!run->active || run->final_sent || run->quarantined ||
      signal.plan_identity != run->request.plan_identity ||
      signal.token != run->request.token ||
      signal.generation != run->request.generation ||
      signal.epoch >= run->request.epoch_count) {
    return Invalid();
  }
  const std::size_t role =
      static_cast<std::size_t>(signal.epoch % ResidencyScheduleRoleCapacity);
  const BackendResidencyScheduleRole &source = run->request.roles[role];
  VulkanResidencyScheduleCell &cell = run->cells[role];
  std::uint32_t expected_generation = 0u;
  if (source.prepared.get() != pipeline || source.bank != signal.bank ||
      !ControlGeneration(source, signal.epoch, expected_generation) ||
      signal.control_generation != expected_generation || cell.signaled ||
      (signal.epoch >= 2u && run->release_count <= signal.epoch - 2u)) {
    return Invalid();
  }
  const std::size_t count = LocalCount(*run, signal.epoch);
  const VulkanTimelinePoint ready = Point(*run, signal.epoch);
  const rund::AccelCheck admitted =
      CanSignalVulkanTimelineReady(pipeline->adapter->timeline, ready);
  if (!admitted.ok) {
    return admitted;
  }
  if (!SelectVulkanResidencyArguments(
          *pipeline,
          std::span<const std::uint32_t>{source.locals.data(), count},
          signal.admission.ok) ||
      !SeedVulkanResidencyControl(*pipeline, signal, signal.admission.ok)) {
    return Unavailable();
  }
  const rund::AccelCheck signaled =
      SignalVulkanTimelineReady(pipeline->adapter->timeline, ready);
  if (!signaled.ok) {
    return signaled;
  }
  cell.admission = signal.admission;
  cell.epoch = signal.epoch;
  cell.signaled = true;
  ++run->signaled_count;
  run->inflight_peak =
      std::max(run->inflight_peak, run->signaled_count - run->release_count);
  run->ready.notify_one();
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
