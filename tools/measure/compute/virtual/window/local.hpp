#pragma once

#include "../../model.hpp"
#include "../backing.hpp"
#include "../model.hpp"

#include <rund/compute/telemetry.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::measure::compute::virtual_window {

using Pipeline = ::rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;
using Profile = ::rund::compute::telemetry::Profile;

inline constexpr std::size_t PageElements = 4'096u;
inline constexpr std::uint32_t FrameCapacity = 3u;
inline constexpr std::size_t LogicalCapacity = PageElements * 12u - 1u;
inline constexpr std::size_t WarmSamples = 60u;
inline constexpr std::array<std::size_t, 3u> ActiveCounts{
    PageElements * 6u - 1u, PageElements * 9u - 1u, PageElements * 12u - 1u};
inline constexpr std::array<std::uint64_t, 3u> EpochCounts{2u, 3u, 4u};
inline constexpr std::size_t FrameBytes = PageElements * sizeof(std::int32_t);
inline constexpr std::size_t ResidentBytes = FrameBytes * FrameCapacity * 4u;

static_assert(ActiveCounts.back() == LogicalCapacity);
static_assert(WarmSamples % 2u == 0u);

struct Prepared final {
  Backend backend;
  std::shared_ptr<virtual_residency::MemoryBacking> input;
  std::shared_ptr<virtual_residency::MemoryBacking> output;
  Pipeline pipeline;
  ::rund::compute::PipelinePlan plan;
  std::vector<std::int32_t> observed;

  Prepared(Backend selected,
           std::shared_ptr<virtual_residency::MemoryBacking> input_owner,
           std::shared_ptr<virtual_residency::MemoryBacking> output_owner,
           Pipeline value) noexcept;
};

[[nodiscard]] std::int32_t seed_value(std::size_t) noexcept;
void seed(std::span<std::int32_t>) noexcept;
[[nodiscard]] std::int32_t expected_value(std::size_t) noexcept;

[[nodiscard]] std::unique_ptr<Prepared> prepare(Backend) noexcept;
[[nodiscard]] bool measure_cell(Prepared &, Prepared &, std::size_t,
                                std::uint64_t) noexcept;

} // namespace rund::measure::compute::virtual_window
