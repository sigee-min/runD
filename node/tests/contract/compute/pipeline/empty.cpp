#include "local.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckZeroWork(rund::compute::Device &device,
                                const Backend backend) {
  using namespace rund::compute;
  auto program = on(device)
                     .map<std::int32_t>("pipeline-empty-work", 0u,
                                        [](auto value) { return value + 1; })
                     .compile();
  const std::array<std::int32_t, 0u> empty{};
  auto input = Upload(device, empty);
  auto output = device.buffer<std::int32_t>(0u);
  if (!program || !input || !output) {
    return 1;
  }
  auto prepared =
      pipeline(device).then(*program, read(*input), write(*output)).prepare();
  if (!prepared || !prepared->run()) {
    return 2;
  }
  std::array<std::int32_t, 0u> observed{};
  const Stats stats = prepared->stats();
  return ReadExact(*prepared, *output, observed) && stats.dispatches == 0u &&
                 stats.command_submits == 0u &&
                 stats.pipeline.verified_step_count == 1u &&
                 stats.backend == backend
             ? 0
             : 3;
}

} // namespace rund_node_test_pipeline
