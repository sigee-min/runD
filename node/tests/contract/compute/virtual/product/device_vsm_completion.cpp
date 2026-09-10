#include "local.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/run/device_vsm/operations.hpp"

#include <memory>
#include <thread>

namespace rund_node_test_virtual::product {
namespace {

namespace vsm = rund::compute::detail::device_vsm_product_detail;
namespace accel = rund::node::accel::detail;

struct Wake final {
  std::unique_ptr<vsm::DeviceVsmProductRun> *run{};
  std::weak_ptr<vsm::DeviceVsmProductOwner> owner{};
  bool published{};
  bool retained{};
  unsigned calls{};
};

void destroy_run(void *const raw) noexcept {
  auto &wake = *static_cast<Wake *>(raw);
  ++wake.calls;
  const auto &run = *wake.run;
  wake.published = run != nullptr && run->owner != nullptr &&
                   run->owner->done.load(std::memory_order_acquire) &&
                   run->callback_count.load(std::memory_order_acquire) == 1u &&
                   run->final.check.ok;
  wake.run->reset();
  wake.retained = !wake.owner.expired();
}

[[nodiscard]] accel::DeviceVsmFinal success() noexcept {
  accel::DeviceVsmFinal result{};
  result.check = {true, "ok"};
  result.terminal = accel::DeviceVsmTerminal::Known;
  return result;
}

} // namespace

int CheckProductDeviceVsmCompletionLifetime() {
  // A wake may consume the result and destroy the run inline. The callback
  // must retain the publication address through that wake without a cycle.
  auto run = std::make_unique<vsm::DeviceVsmProductRun>();
  run->owner = std::make_shared<vsm::DeviceVsmProductOwner>();
  Wake wake{.run = &run, .owner = run->owner};
  run->wake = destroy_run;
  run->wake_user = &wake;
  vsm::complete(run.get(), success());
  if (run != nullptr || wake.calls != 1u || !wake.published ||
      !wake.retained || !wake.owner.expired()) {
    return 1;
  }

  // Synchronous consumers may release their stack-equivalent run immediately
  // after the acquire. Joining occurs only after destruction, never before it.
  for (unsigned iteration = 0u; iteration < 256u; ++iteration) {
    const auto owner = std::make_shared<vsm::DeviceVsmProductOwner>();
    run = std::make_unique<vsm::DeviceVsmProductRun>();
    run->owner = owner;
    std::thread callback{[raw = run.get()] { vsm::complete(raw, success()); }};
    owner->done.wait(false, std::memory_order_acquire);
    const bool published =
        run->final.check.ok &&
        run->callback_count.load(std::memory_order_acquire) == 1u;
    run.reset();
    callback.join();
    if (!published) {
      return 2;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
