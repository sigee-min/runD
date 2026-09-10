#include "internal.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_wide_host {
namespace {

[[nodiscard]] constexpr std::uint64_t
stage_value(const std::uint64_t value) noexcept {
  constexpr std::uint64_t LiteralSum =
      StageLeafCount * (StageLeafCount + 1u) / 2u;
  return value * StageLeafCount + LiteralSum;
}

[[nodiscard]] constexpr std::uint64_t
input_value(const std::uint64_t value, const std::uint64_t first) noexcept {
  constexpr std::uint64_t LiteralSum =
      InputLeafCount * (InputLeafCount + 1u) / 2u;
  return value * InputLeafCount + LiteralSum + (first - 1u) * InputLeafCount;
}

} // namespace

Preparation prepare_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_program(device);
  if (!program) {
    std::fprintf(stderr, "Graph wide compile reason=%u\n",
                 static_cast<unsigned>(program.reason()));
    return {.reason = 1};
  }
  if (!validate_program(*program)) {
    return {.reason = 1};
  }
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  constexpr std::size_t LogicalBytes = ElementCount * sizeof(std::uint64_t);
  constexpr std::size_t PageBytes = FrameElements * sizeof(std::uint64_t);
  std::array<std::vector<std::uint64_t>, InputCount> inputs{};
  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> backings{};
  std::vector<std::uint64_t> expected(ElementCount, 0u);
  for (std::size_t input = 0u; input < InputCount; ++input) {
    inputs[input].resize(ElementCount);
    backings[input] =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
    for (std::size_t index = 0u; index < ElementCount; ++index) {
      inputs[input][index] = 1u + input * 17u + index * (input + 3u);
      expected[index] +=
          input_value(inputs[input][index], input * InputLeafCount + 1u);
    }
    if (backings[input] == nullptr ||
        !backings[input]->seed(std::as_bytes(std::span{inputs[input]}))) {
      return {.reason = 2};
    }
  }
  for (std::uint64_t &value : expected) {
    value = stage_value(value);
  }
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
  auto a = virtual_buffer<std::uint64_t>(ElementCount, backings[0u]);
  auto b = virtual_buffer<std::uint64_t>(ElementCount, backings[1u]);
  auto c = virtual_buffer<std::uint64_t>(ElementCount, backings[2u]);
  auto d = virtual_buffer<std::uint64_t>(ElementCount, backings[3u]);
  auto e = virtual_buffer<std::uint64_t>(ElementCount, backings[4u]);
  auto f = virtual_buffer<std::uint64_t>(ElementCount, backings[5u]);
  auto g = virtual_buffer<std::uint64_t>(ElementCount, backings[6u]);
  auto output = virtual_buffer<std::uint64_t>(ElementCount, output_backing);
  if (!a || !b || !c || !d || !e || !f || !g || !output) {
    return {.reason = 3};
  }
  auto prepared = virtual_pipeline(*program, *a, *b, *c, *d, *e, *f, *g,
                                   *output, ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr, "Graph wide prepare reason=%u\n",
                 static_cast<unsigned>(prepared.reason()));
    return {.reason = 4};
  }
  Pipeline pipeline = std::move(prepared).value();
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<Case>(Case{
              .expected = std::move(expected),
              .input_backings = std::move(backings),
              .output_backing = std::move(output_backing),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

} // namespace rund_node_test_virtual::product::graph_pointwise_wide_host
