#include "local.hpp"

#include "../model.hpp"

#include "../../../../target/selection.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>

namespace rund_node_test_virtual::product {

int CheckProductPipelineConcurrency(const rund::compute::Backend backend) {
  using namespace concurrency;
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-concurrency-owner", PageElements,
                             [](auto value) { return value + 1; })
          .compile();
  if (!program) {
    return 2;
  }

  auto probe = std::make_shared<CallbackProbe>();
  auto input_backing = std::make_shared<ConcurrentBacking>(
      ElementPageBytes, probe, std::byte{0x11});
  auto output_backing =
      std::make_shared<ConcurrentBacking>(ElementPageBytes, probe, TailPoison);
  auto input = virtual_buffer<std::int32_t>(PageElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(PageElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(
                *program, *input, *output,
                ResidencyConfig{.device_resident_bytes = ElementPageBytes * 4u,
                                .host_staging_bytes = ElementPageBytes * 4u})
          : Result<VirtualMap>::fail(Reason::PipelineInvalid);
  if (!prepared) {
    return 3;
  }
  auto pipeline = std::make_shared<VirtualMap>(std::move(*prepared));

  probe->arm_first_callback();
  auto first = std::make_shared<Completion>();
  std::thread first_worker{
      [pipeline, first] { first->publish(pipeline->run()); }};
  if (!probe->wait_until_entered()) {
    probe->release();
    const bool completed = first->wait();
    finish_worker(first_worker, completed);
    return 4;
  }

  const std::uint32_t callbacks_before = probe->callbacks();
  auto second = std::make_shared<Completion>();
  std::thread second_worker{
      [pipeline, second] { second->publish(pipeline->run()); }};
  const bool second_completed_while_blocked = second->wait();
  const Status busy = second->status();
  const std::uint32_t callbacks_after = probe->callbacks();

  probe->release();
  const bool first_completed = first->wait();
  const bool second_completed =
      second_completed_while_blocked || second->wait();
  finish_worker(first_worker, first_completed);
  finish_worker(second_worker, second_completed);

  if (!second_completed_while_blocked || busy ||
      busy.reason() != Reason::PipelineBusy ||
      callbacks_before != callbacks_after) {
    return 5;
  }
  return first_completed && first->status() ? 0 : 6;
}

} // namespace rund_node_test_virtual::product
