#include "local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>

namespace rund_node_bounded_contract {

int CheckVulkanPartitionPipelineWidths() {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(Backend::Vulkan));
  if (!device) {
    return 1;
  }
  const std::array<std::int64_t, 5u> cross_width_input{5, 1, 4, 2, 3};
  auto cross_width =
      on(*device)
          .map<std::int64_t>("partition-cross-width", cross_width_input.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 1; })
          .argsort()
          .filter([](auto index) { return index > 1u; })
          .compile();
  if (!cross_width) {
    return 2;
  }
  auto cross_width_job = cross_width->resident(cross_width_input);
  if (!cross_width_job || !cross_width_job->run()) {
    return 3;
  }
  auto cross_width_output = cross_width_job->read();
  if (!cross_width_output ||
      *cross_width_output != std::vector<std::uint32_t>{2u, 3u}) {
    return 4;
  }

  const std::array<std::uint32_t, 5u> narrow_input{2u, 0u, 1u, 3u, 0u};
  auto narrow = on(*device)
                    .map<std::uint32_t>("partition-narrow", narrow_input.size(),
                                        [](auto value) { return value; })
                    .filter([](auto value) { return value != 0u; })
                    .compile();
  if (!narrow) {
    return 5;
  }
  auto narrow_job = narrow->resident(narrow_input);
  if (!narrow_job || !narrow_job->run()) {
    return 6;
  }
  auto narrow_output = narrow_job->read();
  return narrow_output &&
                 *narrow_output == std::vector<std::uint32_t>{2u, 1u, 3u}
             ? 0
             : 7;
}

} // namespace rund_node_bounded_contract
