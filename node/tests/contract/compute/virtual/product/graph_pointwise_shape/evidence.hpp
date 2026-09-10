#pragma once

#include "../backing.hpp"

#include <rund/compute.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {
struct VirtualPipelineState;
}

namespace rund_node_test_virtual::product::graph_pointwise_shape {

struct EvidenceView final {
  const char *label{};
  std::size_t input_count{};
  std::size_t stage_count{};
  std::size_t frame_elements{};
  std::size_t page_count{};
  std::size_t tail_elements{};
  std::uint64_t frame_capacity{};
  rund::compute::Stats stats{};
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
  std::span<const std::shared_ptr<MemoryVirtualBacking>> input_backings;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  std::span<const std::uint64_t> expected;
};

void stage_generations(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &,
    std::span<std::uint64_t>) noexcept;

[[nodiscard]] bool validate(const EvidenceView &, rund::compute::Backend,
                            const rund::compute::Status &, std::uint64_t,
                            std::span<const std::uint64_t>) noexcept;

} // namespace rund_node_test_virtual::product::graph_pointwise_shape
