#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityCapacity() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  const std::array region_a{authority_detail::RegionA()};
  Authority pipeline;
  std::array<std::uint32_t, 4u> bases{};
  for (std::uint32_t &base : bases) {
    if (!pipeline.register_frames(FrameTier::Device, FrameRole::Input, 1u,
                                  base)) {
      return 39;
    }
  }
  std::array<std::uint64_t, 4u> tokens{};
  for (std::size_t index = 0u; index < bases.size(); ++index) {
    const std::array use{CacheUse{
        .key = {.backing = 30u + index, .version = 1u, .page = 0u},
        .access = Access::Read,
        .next_use = never,
    }};
    const AuthorityResult lease = pipeline.begin(
        use, FrameTier::Device, FrameRole::Input, bases[index], 1u);
    if (!lease || !pipeline.activate(lease.lease.token)) {
      return 40;
    }
    tokens[index] = lease.lease.token;
  }
  if (pipeline
          .begin(region_a, FrameTier::Device, FrameRole::Input, bases[0], 1u)
          .failure != AuthorityFailure::Busy) {
    return 41;
  }
  for (const std::uint64_t token : tokens) {
    if (!pipeline.complete(token, true)) {
      return 42;
    }
  }
  const std::array pipeline_regions{
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = bases[0],
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = bases[1],
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = bases[2],
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = bases[3],
                  .count = 1u},
  };
  if (!pipeline.release_frames(pipeline_regions)) {
    return 42;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
