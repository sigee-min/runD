#include "local.hpp"

#include "src/compute/virtual/run/projection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund_node_test_pipeline_residency {
namespace {

[[nodiscard]] bool
region_matches(const rund::compute::detail::residency::FrameRegion &region,
               const std::uint32_t first) noexcept {
  using namespace rund::compute::detail::residency;
  return region.tier == FrameTier::Host && region.role == FrameRole::Input &&
         region.first == first && region.count == 3u;
}

} // namespace

int CheckGraphHostInputLayout() {
  using namespace rund::compute::detail;

  constexpr std::size_t page_bytes = 16u;
  constexpr std::size_t bank_frames = 6u;
  std::array<std::byte, bank_frames * page_bytes> bank0{};
  std::array<std::byte, bank_frames * page_bytes> bank1{};
  const VirtualRunProjection run{
      .input_page_bytes = page_bytes,
      .first_host_input_frame = 100u,
      .host_input_count = 2u,
      .host_frame_capacity = 3u,
      .input_host_banks = {bank0.data(), bank1.data()},
  };

  const auto input0_bank0 = virtual_graph_host_input_region(run, 0u, 0u);
  const auto input1_bank0 = virtual_graph_host_input_region(run, 1u, 0u);
  const auto input0_bank1 = virtual_graph_host_input_region(run, 0u, 1u);
  const auto input1_bank1 = virtual_graph_host_input_region(run, 1u, 1u);
  if (!region_matches(input0_bank0, 100u) ||
      !region_matches(input1_bank0, 103u) ||
      !region_matches(input0_bank1, 106u) ||
      !region_matches(input1_bank1, 109u)) {
    return 1;
  }

  for (std::uint32_t frame = 100u; frame < 112u; ++frame) {
    const std::uint32_t relative = frame - 100u;
    const std::size_t bank = relative / bank_frames;
    const std::size_t local = relative % bank_frames;
    std::byte *const expected =
        (bank == 0u ? bank0.data() : bank1.data()) + local * page_bytes;
    if (virtual_host_input_frame(run, frame) != expected) {
      return 2;
    }
  }
  if (virtual_host_input_frame(run, 99u) != nullptr ||
      virtual_host_input_frame(run, 112u) != nullptr ||
      virtual_graph_host_input_region(run, 2u, 0u).count != 0u ||
      virtual_graph_host_input_region(run, 0u, 2u).count != 0u) {
    return 3;
  }

  VirtualRunProjection overflow = run;
  overflow.first_host_input_frame =
      std::numeric_limits<std::uint32_t>::max() - 1u;
  if (virtual_graph_host_input_region(overflow, 1u, 1u).count != 0u) {
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
