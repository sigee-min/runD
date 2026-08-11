#pragma once

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace package_compute {
namespace virtual_detail {

constexpr std::size_t logical_elements = 67u;

class Backing final : public rund::compute::VirtualBacking {
public:
  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] rund::compute::Status
  read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept override {
    if (!contains(offset, output.size())) {
      return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
    }
    if (!output.empty()) {
      std::memcpy(output.data(), bytes_.data() + offset, output.size());
    }
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    if (!contains(offset, input.size())) {
      return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
    }
    if (!input.empty()) {
      std::memcpy(bytes_.data() + offset, input.data(), input.size());
    }
    return rund::compute::Status::success();
  }

  void seed(const std::span<const std::int32_t> values) noexcept {
    std::memcpy(bytes_.data(), values.data(), bytes_.size());
  }

  [[nodiscard]] std::array<std::int32_t, logical_elements>
  values() const noexcept {
    std::array<std::int32_t, logical_elements> result{};
    std::memcpy(result.data(), bytes_.data(), bytes_.size());
    return result;
  }

private:
  [[nodiscard]] bool contains(const std::uint64_t offset,
                              const std::size_t bytes) const noexcept {
    return offset <= bytes_.size() && bytes <= bytes_.size() - offset;
  }

  std::array<std::byte, logical_elements * sizeof(std::int32_t)> bytes_{};
};

} // namespace virtual_detail

inline int VirtualResidency() {
  using namespace rund::compute;
  constexpr std::size_t page_elements = 16u;
  constexpr std::size_t frame_capacity = 2u;
  constexpr std::size_t page_count = 5u;
  constexpr std::size_t active_count = 35u;
  std::array<std::int32_t, virtual_detail::logical_elements> input_values{};
  for (std::size_t index = 0u; index < input_values.size(); ++index) {
    input_values[index] = static_cast<std::int32_t>(index) - 31;
  }

  auto device = open(Target::cpu(1u));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .map<std::int32_t>("package-virtual-residency", page_elements,
                             [](auto value) { return value * 3 + 7; })
          .compile();
  if (!program) {
    return 2;
  }

  auto input_backing = std::make_shared<virtual_detail::Backing>();
  auto output_backing = std::make_shared<virtual_detail::Backing>();
  if (input_backing->tier() != VirtualBackingTier::Host ||
      input_backing->max_parallel_reads() != 1u) {
    return 3;
  }
  input_backing->seed(input_values);
  auto input = virtual_buffer<std::int32_t>(input_values.size(), input_backing);
  auto output =
      virtual_buffer<std::int32_t>(input_values.size(), output_backing);
  if (!input || !output) {
    return 4;
  }
  auto prepared = virtual_pipeline(
      *program, *input, *output,
      ResidencyConfig{.device_resident_bytes = frame_capacity * page_elements *
                                               sizeof(std::int32_t) * 4u,
                      .host_staging_bytes = frame_capacity * page_elements *
                                            sizeof(std::int32_t) * 4u});
  if (!prepared) {
    return 5;
  }
  const Status executed = prepared->run(active_count);
  if (!executed) {
    std::fprintf(stderr, "package virtual run failed: %s\n",
                 executed.error().data());
    return 6;
  }

  const auto observed = output_backing->values();
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    const std::int32_t expected =
        index < active_count ? input_values[index] * 3 + 7 : 0;
    if (observed[index] != expected) {
      return 7;
    }
  }

  const PipelinePlan plan = prepared->plan();
  const Stats stats = prepared->stats();
  const MemoryStats memory = prepared->memory();
  const auto profile = prepared->profile();
  const ResidencyStats &residency = stats.pipeline.residency;
  constexpr std::uint64_t logical_bytes =
      input_values.size() * sizeof(std::int32_t) * 2u;
  constexpr std::uint64_t paired_page_bytes =
      page_elements * sizeof(std::int32_t) * 2u;
  return plan.residency.logical_bytes == logical_bytes &&
                 plan.residency.page_bytes == paired_page_bytes &&
                 plan.residency.page_count == page_count &&
                 plan.residency.frame_capacity == frame_capacity &&
                 plan.residency.epoch_count == 3u &&
                 residency.logical_bytes == logical_bytes &&
                 residency.active_count == active_count &&
                 residency.page_bytes == paired_page_bytes &&
                 residency.page_count == page_count &&
                 residency.frame_capacity == frame_capacity &&
                 residency.epoch_count == 2u && residency.page_in_count == 3u &&
                 residency.page_out_count == 3u && stats.output_hash != 0u &&
                 memory.available() && profile &&
                 profile->execution().output_hash == stats.output_hash
             ? 0
             : 8;
}

} // namespace package_compute
