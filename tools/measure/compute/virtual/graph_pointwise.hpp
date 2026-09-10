#pragma once

#include "../model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "src/hash/fnv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::measure::compute::virtual_graph_pointwise {
namespace detail {

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t left = Count / 2u;
    return add_stage<First, left>(value) +
           add_stage<First + left, Count - left>(value);
  }
}

} // namespace detail

struct Spec final {
  static constexpr std::size_t FrameElements = 16u;
  static constexpr std::size_t TailElements = 7u;
  static constexpr std::size_t PageCount = 5u;
  static constexpr std::size_t ElementCount =
      PageCount * FrameElements - TailElements;
  static constexpr std::size_t InputCount = 3u;
  static constexpr std::size_t StageCount = 3u;
  static constexpr std::size_t StageLeafCount = 180u;
  static constexpr std::size_t FrameCapacity = 2u;
  static constexpr std::size_t BatchCount =
      (PageCount + FrameCapacity - 1u) / FrameCapacity;
  static constexpr std::size_t MapRows = 2u;
  static constexpr std::size_t MapEntryCount = 4u;

  using Program = ::rund::compute::Program<std::uint64_t(
      std::uint64_t, std::uint64_t, std::uint64_t)>;

  [[nodiscard]] static ::rund::compute::Result<Program>
  build(const ::rund::compute::Device &device) {
    return ::rund::compute::on(device)
        .input<std::uint64_t>(FrameElements)
        .zip_input<std::uint64_t>(FrameElements)
        .zip_input<std::uint64_t>(FrameElements)
        .branch([](auto first, auto second, auto third) {
          const auto prefix =
              first.map("measure-graph-pointwise-prefix", [](auto value) {
                return detail::add_stage<1u, StageLeafCount>(value);
              });
          const auto middle =
              ::rund::compute::zip(prefix, second)
                  .map("measure-graph-pointwise-middle", [](auto prior,
                                                            auto later) {
                    return prior + detail::add_stage<StageLeafCount + 1u,
                                                     StageLeafCount>(later);
                  });
          return ::rund::compute::zip(middle, third)
              .map("measure-graph-pointwise-output",
                   [](auto prior, auto later) {
                     return prior + detail::add_stage<2u * StageLeafCount + 1u,
                                                      StageLeafCount>(later);
                   });
        })
        .compile();
  }

  [[nodiscard]] static constexpr std::uint64_t
  input_value(const std::size_t input, const std::size_t index) noexcept {
    constexpr std::array<std::uint64_t, InputCount> seeds{5u, 11u, 13u};
    constexpr std::array<std::uint64_t, InputCount> steps{17u, 29u, 43u};
    return input < InputCount ? index * steps[input] + seeds[input] : 0u;
  }

  [[nodiscard]] static constexpr std::size_t
  mapped_page(const std::size_t index) noexcept {
    const std::size_t page = index / FrameElements;
    const std::size_t local = index % FrameElements;
    const std::size_t base = page / FrameCapacity * FrameCapacity;
    const std::size_t count = std::min(FrameCapacity, PageCount - base);
    const std::size_t source = base + count - 1u - (page - base);
    return source * FrameElements + local;
  }

  [[nodiscard]] static constexpr std::uint64_t
  stage_value(const std::uint64_t value, const std::uint64_t first) noexcept {
    constexpr std::uint64_t literal_sum =
        StageLeafCount * (StageLeafCount + 1u) / 2u;
    return value * StageLeafCount + literal_sum + (first - 1u) * StageLeafCount;
  }

  [[nodiscard]] static constexpr std::uint64_t
  expected_value(const std::size_t index) noexcept {
    const std::uint64_t first = stage_value(input_value(0u, index), 1u);
    const std::uint64_t second =
        stage_value(input_value(1u, mapped_page(index)), StageLeafCount + 1u);
    const std::uint64_t third =
        stage_value(input_value(2u, index), 2u * StageLeafCount + 1u);
    return first + second + third;
  }

  [[nodiscard]] static constexpr std::array<::rund::compute::GraphPageMapEntry,
                                            MapEntryCount>
  page_map_entries() noexcept {
    return {
        ::rund::compute::GraphPageMapEntry{
            .input = 0u,
            .target_local = 0u,
            .source_local = 0u,
            .origin = ::rund::compute::PageOrigin::Begin},
        ::rund::compute::GraphPageMapEntry{
            .input = 0u,
            .target_local = 1u,
            .source_local = 1u,
            .origin = ::rund::compute::PageOrigin::Begin},
        ::rund::compute::GraphPageMapEntry{
            .input = 1u,
            .target_local = 0u,
            .source_local = 0u,
            .origin = ::rund::compute::PageOrigin::End},
        ::rund::compute::GraphPageMapEntry{
            .input = 1u,
            .target_local = 1u,
            .source_local = 1u,
            .origin = ::rund::compute::PageOrigin::End},
    };
  }

  [[nodiscard]] static std::uint64_t
  hash(const std::span<const std::uint64_t> values) noexcept {
    return ::rund::node::hash_detail::HashBytes(
        values.data(), values.size() * sizeof(std::uint64_t));
  }

  [[nodiscard]] static std::uint64_t expected_hash() noexcept {
    std::array<std::uint64_t, ElementCount> values{};
    for (std::size_t index = 0u; index < values.size(); ++index) {
      values[index] = expected_value(index);
    }
    return hash(std::span{values});
  }

  [[nodiscard]] static constexpr std::uint64_t input_digest() noexcept {
    std::uint64_t result = 1469598103934665603ull;
    for (std::size_t input = 0u; input < InputCount; ++input) {
      for (std::size_t index = 0u; index < ElementCount; ++index) {
        result ^= input_value(input, index);
        result *= 1099511628211ull;
      }
    }
    return result == 0u ? 1u : result;
  }
};

[[nodiscard]] bool Measure(Backend backend);

} // namespace rund::measure::compute::virtual_graph_pointwise
