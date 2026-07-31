#include "local.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_bounded_contract {

int CheckTypedBoundedMap(const rund::compute::Target target) {
  using namespace rund::compute;
  const std::array<std::int64_t, 4u> input{1, 2, 3, 4};
  auto output = on(target, input)
                    .filter([](auto value) { return value > 2; })
                    .map("bounded-mask",
                         [](auto value) {
                           return select<std::uint64_t>(value > 3, 1u, 0u);
                         })
                    .collect();
  return output && *output == std::vector<std::uint64_t>{0u, 1u} ? 0 : 1;
}

int CheckInactiveTail(const rund::compute::Target target) {
  using namespace rund::compute;
  const std::array<std::int32_t, 2u> input{
      1, std::numeric_limits<std::int32_t>::max()};
  auto output = on(target, input)
                    .filter([](auto value) { return value < 2; })
                    .reduce(Reduce::Sum)
                    .collect();
  return output && *output == std::vector<std::int32_t>{1} ? 0 : 1;
}

int CheckReduceRewrite(const rund::compute::Target target) {
  using namespace rund::compute;
  const std::array<std::int64_t, 4u> input{1, 2, 3, 4};
  auto program =
      on(target)
          .template map<std::int64_t>("bounded-reduce-rewrite", input.size(),
                                      [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .reduce(Reduce::Sum)
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  auto output = job->read();
  if (!output || *output != std::vector<std::int64_t>{7}) {
    return 3;
  }
  const std::array<std::int64_t, 4u> rewritten{4, 1, 1, 1};
  if (!job->write(rewritten) || !job->run() ||
      job->stats().download_events != 0u) {
    return 4;
  }
  output = job->read();
  return output && *output == std::vector<std::int64_t>{4} ? 0 : 5;
}

} // namespace rund_node_bounded_contract
