#include "local.hpp"

#include "backing.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

constexpr std::size_t FrameElements = 16u;
constexpr std::size_t Radius = 2u;
constexpr std::size_t PayloadElements = FrameElements - Radius * 2u;
constexpr std::size_t WindowElements = 53u;
constexpr std::size_t WindowPages =
    (WindowElements + PayloadElements - 1u) / PayloadElements;

[[nodiscard]] std::int32_t value(const std::size_t index) noexcept {
  return static_cast<std::int32_t>((index * 17u + 11u) % 41u);
}

[[nodiscard]] std::int32_t expected(const std::size_t index) noexcept {
  std::int32_t total = 0;
  for (std::size_t sample = 0u; sample < Radius * 2u + 1u; ++sample) {
    const std::size_t raw = index + sample;
    const std::size_t selected = raw < Radius ? 0u
                                 : raw - Radius >= WindowElements
                                     ? WindowElements - 1u
                                     : raw - Radius;
    total += value(selected);
  }
  return total;
}

[[nodiscard]] std::int32_t expected_clip_min(const std::size_t index) noexcept {
  std::int32_t minimum = std::numeric_limits<std::int32_t>::max();
  for (std::size_t sample = 0u; sample < Radius * 2u + 1u; ++sample) {
    const std::size_t raw = index + sample;
    if (raw >= Radius && raw - Radius < WindowElements) {
      minimum = std::min(minimum, value(raw - Radius));
    }
  }
  return minimum;
}

} // namespace

int CheckProductWindow(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .map<std::int32_t>("virtual-product-window", FrameElements,
                             [](auto input) { return input; })
          .window(WindowSpec{
              .op = Window::Sum, .radius = Radius, .edge = WindowEdge::Clamp})
          .compile();
  if (!program || program->graph().nodes.empty() ||
      program->graph().nodes.back().footprint.pattern !=
          graph::AccessPattern::Window) {
    std::fprintf(stderr, "virtual window program ok=%u nodes=%zu pattern=%u\n",
                 static_cast<unsigned>(static_cast<bool>(program)),
                 program ? program->graph().nodes.size() : 0u,
                 program && !program->graph().nodes.empty()
                     ? static_cast<unsigned>(
                           program->graph().nodes.back().footprint.pattern)
                     : 255u);
    return 2;
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  std::array<std::int32_t, WindowElements> input_values{};
  for (std::size_t index = 0u; index < input_values.size(); ++index) {
    input_values[index] = value(index);
  }
  if (!input_backing->seed(std::as_bytes(std::span{input_values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(WindowElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(WindowElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || !prepared->run()) {
    return 4;
  }
  std::array<std::byte, WindowElements * sizeof(std::int32_t)> observed{};
  if (!output_backing->observe(observed)) {
    return 5;
  }
  for (std::size_t index = 0u; index < WindowElements; ++index) {
    std::int32_t actual = 0;
    std::memcpy(&actual, observed.data() + index * sizeof(actual),
                sizeof(actual));
    if (actual != expected(index)) {
      return 6;
    }
  }
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  constexpr std::uint64_t ExpectedBackingElements = 14u + 16u * 3u + 7u;
  if (prepared->plan().residency.page_count != WindowPages ||
      residency.page_count != WindowPages || residency.frame_capacity != 2u ||
      residency.page_in_count != WindowPages ||
      residency.late_page_count != 2u ||
      residency.prefetch_count != WindowPages - 2u ||
      residency.backing_read_bytes !=
          ExpectedBackingElements * sizeof(std::int32_t) ||
      residency.page_in_bytes != residency.backing_read_bytes ||
      residency.backing_write_bytes != observed.size() ||
      stats.uploaded_bytes !=
          WindowPages * FrameElements * sizeof(std::int32_t) ||
      stats.downloaded_bytes !=
          WindowPages * FrameElements * sizeof(std::int32_t)) {
    return 7;
  }
  auto clip_program =
      on(*device)
          .map<std::int32_t>("virtual-product-window-clip", FrameElements,
                             [](auto item) { return item; })
          .window(WindowSpec{
              .op = Window::Min, .radius = Radius, .edge = WindowEdge::Clip})
          .compile();
  auto clip_backing = std::make_shared<MemoryVirtualBacking>(
      WindowElements * sizeof(std::int32_t),
      FrameElements * sizeof(std::int32_t));
  auto clip_output = virtual_buffer<std::int32_t>(WindowElements, clip_backing);
  auto clip = clip_program && clip_output
                  ? virtual_pipeline(*clip_program, *input, *clip_output,
                                     ResidencyConfig{})
                  : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                        Reason::PipelineInvalid);
  if (!clip || !clip->run() || !clip_backing->observe(observed)) {
    return 8;
  }
  for (std::size_t index = 0u; index < WindowElements; ++index) {
    std::int32_t actual = 0;
    std::memcpy(&actual, observed.data() + index * sizeof(actual),
                sizeof(actual));
    if (actual != expected_clip_min(index)) {
      return 9;
    }
  }
  if (clip->stats().pipeline.residency.page_in_count != WindowPages ||
      clip->stats().pipeline.residency.cache_hit_count != 0u) {
    return 10;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
