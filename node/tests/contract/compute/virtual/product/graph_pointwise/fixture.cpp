#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <limits>

namespace rund_node_test_virtual::product::graph_pointwise {
namespace {

[[nodiscard]] constexpr std::uint64_t
expected_stage(const std::uint64_t value, const std::uint64_t first) noexcept {
  constexpr std::uint64_t LiteralSum =
      StageLeafCount * (StageLeafCount + 1u) / 2u;
  return value * StageLeafCount + LiteralSum + (first - 1u) * StageLeafCount;
}

} // namespace

Preparation prepare_case(const rund::compute::Device &device,
                         const std::size_t page_count) {
  using namespace rund::compute;
  if (page_count == 0u ||
      page_count > std::numeric_limits<std::size_t>::max() / FrameElements) {
    return {.reason = 1};
  }
  auto program = build_program(device);
  if (!program || !validate_program(*program)) {
    return {.reason = 2};
  }
  const std::size_t element_count = page_count * FrameElements - TailElements;
  std::vector<std::uint64_t> input_values(element_count);
  std::vector<std::uint64_t> expected(element_count);
  for (std::size_t index = 0u; index < element_count; ++index) {
    input_values[index] = index * 17u + 5u;
    expected[index] = expected_stage(input_values[index], 1u) +
                      expected_stage(input_values[index], StageLeafCount + 1u);
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      element_count * sizeof(std::uint64_t),
      FrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      element_count * sizeof(std::uint64_t),
      FrameElements * sizeof(std::uint64_t));
  if (!input_backing->seed(std::as_bytes(std::span{input_values}))) {
    return {.reason = 3};
  }
  auto input = virtual_buffer<std::uint64_t>(element_count, input_backing);
  auto output = virtual_buffer<std::uint64_t>(element_count, output_backing);
  auto pipeline =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!pipeline) {
    return {.reason = 4};
  }
  Pipeline product = std::move(pipeline).value();
  const std::shared_ptr<detail::VirtualPipelineState> state =
      detail::VirtualPipelineAccess::state(product);
  return {
      .value = std::make_unique<Case>(Case{
          .page_count = page_count,
          .input_values = std::move(input_values),
          .expected = std::move(expected),
          .input_backing = std::move(input_backing),
          .output_backing = std::move(output_backing),
          .pipeline = std::move(product),
          .state = state,
      }),
  };
}

} // namespace rund_node_test_virtual::product::graph_pointwise
