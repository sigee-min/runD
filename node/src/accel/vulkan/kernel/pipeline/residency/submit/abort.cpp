#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
AbortVulkanResidencyWindow(const std::shared_ptr<void> &prepared,
                           const BackendResidencyWindowAbort &abort) noexcept {
  using vulkan_residency_submit_detail::Invalid;
  auto *const leader = static_cast<VulkanPipeline *>(prepared.get());
  VulkanResidencySelection *const selection =
      leader == nullptr ? nullptr : leader->residency.get();
  VulkanResidencyWindowRun *const run =
      selection == nullptr
          ? nullptr
          : selection->active_window.load(std::memory_order_acquire);
  if (!ValidVulkanPipeline(leader) || selection == nullptr || run == nullptr ||
      abort.failure.ok || abort.failure.reason == nullptr ||
      abort.failure.reason[0u] == '\0' || abort.plan_identity == 0u ||
      abort.token == 0u || abort.generation == 0u) {
    return Invalid();
  }
  {
    std::lock_guard lock{run->gate};
    if (!run->active || run->final_sent || run->aborting ||
        run->request.batch_count == 0u ||
        run->request.batches[0u].prepared.get() != leader ||
        abort.plan_identity != run->request.plan_identity ||
        abort.token != run->request.token ||
        abort.generation != run->request.generation) {
      return Invalid();
    }
    run->abort_failure = abort.failure;
    run->aborting = true;
    run->quarantined = true;
    leader->adapter->residency_quarantined.store(true,
                                                 std::memory_order_release);
  }
  run->ready.notify_all();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
