#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline {
namespace {

struct ValueField final {};
struct WeightField final {};

} // namespace

[[nodiscard]] int CheckSurfaceRecord(
    rund::compute::Device &device,
    const rund::compute::Buffer<std::int32_t> &input) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};

  auto record_program =
      on(device)
          .map<std::int32_t>("pipeline-record", input_values.size(),
                             [](auto value) {
                               return record(field<ValueField>(value * 2),
                                             field<WeightField>(value + 7));
                             })
          .compile();
  auto record_values = device.buffer<std::int32_t>(input_values.size());
  auto record_weights = device.buffer<std::int32_t>(input_values.size());
  if (!record_program || !record_values || !record_weights) {
    return 2;
  }
  auto record_pipeline = pipeline(device)
                             .then(*record_program, read(input),
                                   write(*record_values, *record_weights))
                             .prepare();
  if (!record_pipeline || !record_pipeline->run()) {
    return 3;
  }
  std::array<std::int32_t, 4u> values{};
  std::array<std::int32_t, 4u> weights{};
  if (!ReadExact(*record_pipeline, *record_values, values) ||
      !ReadExact(*record_pipeline, *record_weights, weights) ||
      values != std::array<std::int32_t, 4u>{2, 4, 6, 8} ||
      weights != std::array<std::int32_t, 4u>{8, 9, 10, 11} ||
      record_pipeline->stats().output_hash == 0u) {
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
