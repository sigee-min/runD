#pragma once

#include "../backing.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t TailElements = 7u;
inline constexpr std::size_t StageLeafCount = 255u;
inline constexpr std::size_t StageCount = 3u;

using Program = rund::compute::Program<std::uint64_t(std::uint64_t)>;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(std::uint64_t)>;

struct Case final {
  std::size_t page_count{};
  std::vector<std::uint64_t> input_values;
  std::vector<std::uint64_t> expected;
  std::shared_ptr<MemoryVirtualBacking> input_backing;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value;
  int reason{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &device);
[[nodiscard]] bool validate_program(const Program &program) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &device,
                                       std::size_t page_count);
[[nodiscard]] bool observe_output(Case &test_case) noexcept;
[[nodiscard]] bool validate_success(Case &test_case,
                                    rund::compute::Backend backend,
                                    const rund::compute::Status &status,
                                    std::uint64_t initial_version) noexcept;
[[nodiscard]] std::array<std::uint64_t, StageCount>
stage_generations(const Case &) noexcept;
[[nodiscard]] int check_recovery(const rund::compute::Device &device,
                                 rund::compute::Backend backend);
[[nodiscard]] int check_scale(const rund::compute::Device &device,
                              rund::compute::Backend backend,
                              std::size_t page_count);

} // namespace rund_node_test_virtual::product::graph_pointwise
