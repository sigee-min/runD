#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSurfaceBounded(
    rund::compute::Device &device,
    const rund::compute::Buffer<std::int32_t> &input) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};

  auto bounded_program =
      on(device)
          .map<std::int32_t>("pipeline-bounded", input_values.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .compile();
  auto bounded_values = device.buffer<std::int32_t>(input_values.size());
  auto bounded_count = device.buffer<std::uint32_t>(1u);
  if (!bounded_program || !bounded_values || !bounded_count) {
    return 5;
  }
  auto bounded_pipeline = pipeline(device)
                              .then(*bounded_program, read(input),
                                    write(*bounded_values, *bounded_count))
                              .prepare();
  if (!bounded_pipeline || !bounded_pipeline->run()) {
    return 6;
  }
  std::array<std::int32_t, 4u> compacted{};
  std::array<std::uint32_t, 1u> count{};
  if (!ReadExact(*bounded_pipeline, *bounded_values, compacted) ||
      !ReadExact(*bounded_pipeline, *bounded_count, count) || count[0] != 2u ||
      compacted[0] != 3 || compacted[1] != 4) {
    return 7;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
