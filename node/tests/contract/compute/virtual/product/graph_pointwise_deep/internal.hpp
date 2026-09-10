#pragma once

#include "../backing.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise_deep {

inline constexpr std::size_t InputCount = 6u;
inline constexpr std::size_t StageCount = 3u;
inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t PageCount = 5u;
inline constexpr std::size_t TailElements = 7u;
inline constexpr std::size_t InputLeafCount = 37u;
// Each stage remains below ExpressionCapacity, while either adjacent pair is
// too large to collapse into one Map. This preserves three physical stages.
inline constexpr std::size_t StageLeafCount = 300u;

using Program = rund::compute::Program<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t)>;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t)>;

struct Case final {
  std::vector<std::uint64_t> expected;
  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> input_backings;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value;
  int reason{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] bool validate_program(const Program &) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &);
[[nodiscard]] std::array<std::uint64_t, StageCount>
stage_generations(const Case &) noexcept;
[[nodiscard]] bool
validate_case(Case &, rund::compute::Backend, const rund::compute::Status &,
              std::uint64_t,
              const std::array<std::uint64_t, StageCount> &) noexcept;

} // namespace rund_node_test_virtual::product::graph_pointwise_deep
