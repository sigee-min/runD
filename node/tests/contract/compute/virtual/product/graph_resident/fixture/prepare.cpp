#include "local.hpp"

#include "src/compute/virtual/backing.hpp"

#include <array>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_virtual::product::graph_resident {

Preparation prepare_case(const rund::compute::Device &device,
                         const Variant variant) {
  using namespace rund::compute;
  auto program = build_program(device, variant);
  if (!program || !validate_program(*program, variant)) {
    return {.reason = 1};
  }

  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> expected(ElementCount);
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    for (std::size_t input = 0u; input < InputCount; ++input) {
      values[input].push_back(Workload::input_value(input, index));
    }
    expected[index] = Workload::expected_value(index, variant);
  }

  std::array<std::shared_ptr<VirtualBacking>, InputCount> inputs{};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    auto backing =
        resident_virtual_backing<std::uint64_t>(device, ElementCount);
    if (!backing ||
        !seed_backing(*backing, std::as_bytes(std::span{values[index]}))) {
      return {.reason = 2};
    }
    inputs[index] = std::move(backing).value();
  }
  auto output = resident_virtual_backing<std::uint64_t>(device, ElementCount);
  if (!output) {
    return {.reason = 3};
  }

  auto first = virtual_buffer<std::uint64_t>(ElementCount, inputs[0u]);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, inputs[1u]);
  auto third = virtual_buffer<std::uint64_t>(ElementCount, inputs[2u]);
  auto result = virtual_buffer<std::uint64_t>(ElementCount, output.value());
  if (!first || !second || !third || !result) {
    return {.reason = 4};
  }
  auto prepared = virtual_pipeline(*program, *first, *second, *third, *result,
                                   ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr, "GraphResident prepare reason=%u native=%s\n",
                 static_cast<unsigned>(prepared.reason()),
                 prepared.location().native_reason_key == nullptr
                     ? "none"
                     : prepared.location().native_reason_key);
    return {.reason = 5};
  }
  Pipeline pipeline = std::move(prepared).value();
  if (pipeline.plan().residency.frame_capacity != 2u) {
    return {.reason = 6};
  }
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<Case>(Case{
              .variant = variant,
              .first_values = std::move(values[0u]),
              .second_values = std::move(values[1u]),
              .third_values = std::move(values[2u]),
              .expected = std::move(expected),
              .inputs = std::move(inputs),
              .output = std::move(output).value(),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

} // namespace rund_node_test_virtual::product::graph_resident
