#pragma once

#include "../backing.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise_multi {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t TailElements = 7u;
inline constexpr std::size_t InputCount = 3u;
// Every authored stage stays below the public Graph expression bound, while
// substituting its predecessor into the following stage exceeds the single
// service-free Map bound. This deterministically selects GraphPointwise even
// with a cold compiler cache.
inline constexpr std::size_t StageLeafCount = 180u;
inline constexpr std::size_t StageCount = 3u;
inline constexpr std::size_t FailedLeafCapacity = 32u;
inline constexpr std::size_t OutputNpos =
    std::numeric_limits<std::size_t>::max();

using Program = rund::compute::Program<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;

struct Case final {
  std::size_t page_count{};
  std::vector<std::uint64_t> first_values;
  std::vector<std::uint64_t> second_values;
  std::vector<std::uint64_t> third_values;
  std::vector<std::uint64_t> expected;
  std::shared_ptr<MemoryVirtualBacking> first_backing;
  std::shared_ptr<MemoryVirtualBacking> second_backing;
  std::shared_ptr<MemoryVirtualBacking> third_backing;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value;
  int reason{};
};

struct OutputObservation final {
  bool readable{};
  bool matches{};
  std::size_t first_bad_index{OutputNpos};
  std::uint64_t actual{};
  std::uint64_t expected{};
};

struct SuccessReport final {
  OutputObservation output{};
  std::array<std::uint64_t, StageCount> before_generations{};
  std::array<std::uint64_t, StageCount> after_generations{};
  std::array<const char *, FailedLeafCapacity> failed_names{};
  std::uint64_t failed_mask{};
  std::size_t failed_count{};
  std::uint64_t command_submits{};
  std::uint64_t dispatches{};
  std::uint64_t page_in{};
  std::uint64_t page_out{};
  std::uint64_t write_count{};
  std::uint64_t write_bytes{};
  bool owner{};
  bool output_reached{};
  bool published{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] bool validate_program(const Program &) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &,
                                       std::size_t);
[[nodiscard]] OutputObservation observe_output(Case &) noexcept;
[[nodiscard]] std::array<std::uint64_t, StageCount>
stage_generations(const Case &) noexcept;
[[nodiscard]] bool validate_success(Case &, rund::compute::Backend,
                                    const rund::compute::Status &,
                                    std::uint64_t, bool,
                                    SuccessReport &) noexcept;
[[nodiscard]] bool validate_warm(Case &, rund::compute::Backend,
                                 const rund::compute::Status &, std::uint64_t,
                                 const BackingFacts &,
                                 const std::array<std::uint64_t, StageCount> &,
                                 const std::array<std::uint64_t, StageCount> &,
                                 SuccessReport &) noexcept;
void report_failure(const SuccessReport &, rund::compute::Backend,
                    const rund::compute::Status &, std::size_t,
                    const char *) noexcept;
[[nodiscard]] int check_scale(const rund::compute::Device &,
                              rund::compute::Backend, std::size_t);

} // namespace rund_node_test_virtual::product::graph_pointwise_multi
