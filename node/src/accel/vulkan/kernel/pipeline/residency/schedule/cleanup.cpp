#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace schedule {

void ClearActive(VulkanResidencyScheduleRun &run) noexcept {
  for (std::size_t role = 0u; role < run.request.role_count; ++role) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(run.request.roles[role].prepared.get());
    if (pipeline == nullptr || pipeline->residency == nullptr) {
      continue;
    }
    VulkanResidencyScheduleRun *expected = &run;
    static_cast<void>(
        pipeline->residency->active_schedule.compare_exchange_strong(
            expected, nullptr, std::memory_order_acq_rel,
            std::memory_order_acquire));
  }
}

} // namespace schedule

using namespace schedule;

rund::AccelCheck AbortVulkanResidencySchedule(
    const std::shared_ptr<void> &prepared,
    const BackendResidencyWindowAbort &abort) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  VulkanResidencySelection *const selection =
      pipeline == nullptr ? nullptr : pipeline->residency.get();
  VulkanResidencyScheduleRun *const run =
      selection == nullptr
          ? nullptr
          : selection->active_schedule.load(std::memory_order_acquire);
  if (!ValidVulkanPipeline(pipeline) || run == nullptr || abort.failure.ok ||
      abort.failure.reason == nullptr || abort.plan_identity == 0u ||
      abort.token == 0u || abort.generation == 0u) {
    return Invalid();
  }
  {
    std::lock_guard lock{run->gate};
    if (!run->active || run->final_sent || run->aborting ||
        abort.plan_identity != run->request.plan_identity ||
        abort.token != run->request.token ||
        abort.generation != run->request.generation) {
      return Invalid();
    }
    run->abort_failure = abort.failure;
    run->aborting = true;
    run->quarantined = true;
    pipeline->adapter->residency_quarantined.store(true,
                                                   std::memory_order_release);
  }
  run->ready.notify_all();
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
