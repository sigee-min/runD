#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSurfaceSharedCount(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto shared_count_program =
      on(device)
          .map<std::uint32_t>("pipeline-shared-count", input_values.size(),
                              [](auto value) { return value; })
          .filter([](auto value) { return value > 1u; })
          .branch([](auto values) {
            auto next = values.map("pipeline-shared-next",
                                   [](auto value) { return value + 10u; });
            auto last = values.map("pipeline-shared-last",
                                   [](auto value) { return value + 20u; });
            return outputs(values, next, last);
          })
          .compile();
  constexpr std::array<std::uint32_t, 17u> shared_initial{
      1u, 2u, 3u, 4u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
      0u};
  auto shared_storage = Upload(device, shared_initial);
  if (!shared_count_program || !shared_storage) {
    return 13;
  }
  auto shared_input = shared_storage->view(0u, input_values.size());
  auto first = shared_storage->view(input_values.size(), input_values.size());
  auto shared_count = shared_storage->view(input_values.size() * 2u, 1u);
  auto second =
      shared_storage->view(input_values.size() * 2u + 1u, input_values.size());
  auto last =
      shared_storage->view(input_values.size() * 3u + 1u, input_values.size());
  if (!shared_input || !first || !shared_count || !second || !last) {
    return 14;
  }
  auto shared_count_pipeline =
      pipeline(device)
          .then(*shared_count_program, read(*shared_input),
                write(*first, *shared_count, *second, *shared_count, *last,
                      *shared_count))
          .prepare();
  const Status shared_run = shared_count_pipeline
                                ? shared_count_pipeline->run()
                                : Status::fail(shared_count_pipeline.reason());
  std::array<std::uint32_t, shared_initial.size()> shared_values{};
  if (!shared_run ||
      !ReadExact(*shared_count_pipeline, *shared_storage, shared_values) ||
      shared_values[4] != 2u || shared_values[5] != 3u ||
      shared_values[6] != 4u || shared_values[8] != 3u ||
      shared_values[9] != 12u || shared_values[10] != 13u ||
      shared_values[11] != 14u || shared_values[13] != 22u ||
      shared_values[14] != 23u || shared_values[15] != 24u) {
    return 15;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
