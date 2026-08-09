#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace rund_node_test_pipeline {
namespace {

constexpr std::size_t Capacity = 257u;
constexpr std::size_t Radius = 128u;

[[nodiscard]] constexpr std::uint32_t
Mapped(const std::uint32_t value) noexcept {
  return (value & 3u) + 1u;
}

void Fill(std::vector<std::uint32_t> &values,
          const std::size_t active) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    const std::uint32_t narrow = static_cast<std::uint32_t>(index);
    values[index] =
        index < active
            ? (narrow * 2654435761u) ^ (narrow >> 7u) ^ 0x9e3779b9u
            : (index % 2u == 0u ? 0u
                                : std::numeric_limits<std::uint32_t>::max());
  }
}

[[nodiscard]] std::uint32_t Oracle(const std::span<const std::uint32_t> input,
                                   const std::size_t active) noexcept {
  if (active == 0u) {
    return 0u;
  }
  std::uint64_t total = 0u;
  for (std::size_t index = 0u; index < active; ++index) {
    std::uint32_t minimum = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t offset = 0u; offset != Radius * 2u + 1u; ++offset) {
      const std::size_t raw = index + offset;
      const std::size_t sample = raw < Radius             ? 0u
                                 : raw - Radius >= active ? active - 1u
                                                          : raw - Radius;
      minimum = std::min(minimum, Mapped(input[sample]));
    }
    if ((minimum & 7u) != 0u) {
      total += minimum;
    }
  }
  return static_cast<std::uint32_t>(total);
}

} // namespace

[[nodiscard]] int CheckResidentRangeTransition(rund::compute::Device &device,
                                               const Backend backend) {
  using namespace rund::compute;
  auto program = on(device)
                     .input<Bounded<std::uint32_t>>(Capacity)
                     .map("pipeline-range-map",
                          [](auto value) { return (value & 3u) + 1u; })
                     .window({.op = Window::Min, .radius = Radius})
                     .filter([](auto value) { return (value & 7u) != 0u; })
                     .reduce(Reduce::Sum)
                     .compile();
  std::vector<std::uint32_t> values(Capacity);
  Fill(values, 0u);
  const std::array<std::uint32_t, 1u> zero{0u};
  auto input = device.upload<std::uint32_t>(std::span{values});
  auto count = device.upload<std::uint32_t>(std::span{zero});
  auto output = device.buffer<std::uint32_t>(1u);
  if (!program || !input || !count || !output) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .then(*program, read(*input, *count), write(*output))
                      .prepare();
  if (!prepared) {
    return 2;
  }
  if (!prepared->run()) {
    return 3;
  }
  const Stats first = prepared->stats();
  std::array<std::uint32_t, 1u> cold{};
  if (first.pipeline_compiles != 0u || first.buffer_allocations != 0u ||
      first.descriptor_pool_creations != 0u ||
      first.descriptor_set_allocations != 0u || first.uploaded_bytes != 0u ||
      first.download_events != 0u || first.downloaded_bytes != 0u ||
      first.pipeline.rebinding_count != 0u ||
      first.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
      !prepared->read(*output, std::span<std::uint32_t>{cold}) ||
      cold[0u] != 0u || !prepared->run()) {
    return 3;
  }

  constexpr std::array<std::size_t, 4u> counts{0u, 9u, 129u, Capacity};
  for (std::size_t phase = 0u; phase + 1u < counts.size(); ++phase) {
    const std::size_t current = counts[phase];
    const std::size_t next = counts[phase + 1u];
    const std::uint32_t expected = Oracle(values, current);
    std::vector<std::uint32_t> next_values(Capacity);
    Fill(next_values, next);
    const std::array<std::uint32_t, 1u> next_count{
        static_cast<std::uint32_t>(next)};
    std::size_t callbacks = 0u;
    const Status transitioned = host_feedback(
        *prepared, 2u, [&](HostIteration &iteration) noexcept -> Status {
          ++callbacks;
          if (iteration.completed() != 1u) {
            return Status::success();
          }
          std::array<std::uint32_t, 1u> observed{};
          Status status = iteration.read(*output, std::span{observed});
          if (!status || observed[0u] != expected) {
            return Status::fail(Reason::CompletionInvalid);
          }
          status = iteration.write(*input, std::span{next_values});
          if (!status) {
            return status;
          }
          return iteration.write(*count, std::span{next_count});
        });
    if (!transitioned || callbacks != 2u) {
      return 4 + static_cast<int>(phase);
    }
    values = std::move(next_values);
  }

  std::array<std::uint32_t, 1u> terminal{};
  if (!prepared->read(*output, std::span<std::uint32_t>{terminal}) ||
      terminal[0u] != Oracle(values, Capacity)) {
    return 7;
  }

  std::array<RangeInfo, 1u> ranges{};
  const RangeSnapshot snapshot = program->ranges(ranges);
  return snapshot.written == 1u && snapshot.total == 1u &&
                 !snapshot.truncated() && ranges[0u].execution &&
                 ranges[0u].source
             ? 0
             : 9;
}

} // namespace rund_node_test_pipeline
