#include "local.hpp"

#include "../model.hpp"

#include "../../../../target/selection.hpp"

#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>

namespace rund_node_test_virtual::product {

int CheckProductBackingConcurrency(const rund::compute::Backend backend) {
  using namespace concurrency;
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-concurrency-backing",
                             PageElements, [](auto value) { return value + 1; })
          .compile();
  if (!program) {
    return 2;
  }

  auto probe = std::make_shared<CallbackProbe>();
  auto left_backing = std::make_shared<ConcurrentBacking>(
      ElementPageBytes, probe, std::byte{0x21});
  auto right_backing = std::make_shared<ConcurrentBacking>(
      ElementPageBytes, probe, std::byte{0x42});
  auto left = virtual_buffer<std::int32_t>(PageElements, left_backing);
  auto right = virtual_buffer<std::int32_t>(PageElements, right_backing);
  if (!left || !right) {
    return 3;
  }
  auto forward =
      virtual_pipeline(*program, *left, *right, ResidencyConfig{.slots = 1u});
  auto reverse =
      virtual_pipeline(*program, *right, *left, ResidencyConfig{.slots = 1u});
  if (!forward || !reverse) {
    return 4;
  }

  auto forward_pipeline = std::make_shared<VirtualMap>(std::move(*forward));
  auto reverse_pipeline = std::make_shared<VirtualMap>(std::move(*reverse));
  auto start = std::make_shared<std::barrier<>>(3);
  auto forward_completion = std::make_shared<Completion>();
  auto reverse_completion = std::make_shared<Completion>();
  probe->arm_first_callback();
  std::thread forward_worker{[forward_pipeline, start, forward_completion] {
    start->arrive_and_wait();
    forward_completion->publish(forward_pipeline->run());
  }};
  std::thread reverse_worker{[reverse_pipeline, start, reverse_completion] {
    start->arrive_and_wait();
    reverse_completion->publish(reverse_pipeline->run());
  }};
  start->arrive_and_wait();

  const bool callback_entered = probe->wait_until_entered();
  probe->release();
  const bool forward_completed = forward_completion->wait();
  const bool reverse_completed = reverse_completion->wait();
  finish_worker(forward_worker, forward_completed);
  finish_worker(reverse_worker, reverse_completed);

  if (!callback_entered || !forward_completed || !reverse_completed) {
    return 5;
  }
  if (!forward_completion->status() || !reverse_completion->status()) {
    return 6;
  }
  return probe->max_active() == 1u && probe->callbacks() == 4u ? 0 : 7;
}

} // namespace rund_node_test_virtual::product
