#include "internal.hpp"

#include "../graph_pointwise_shape/stage.hpp"
#include "src/compute/virtual/backing.hpp"

#include <cstdio>
#include <span>
#include <utility>

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

namespace {

struct InputData final {
  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> expected;
};

[[nodiscard]] InputData make_data() {
  using graph_pointwise_shape::stage_value;
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  InputData result{};
  result.expected.assign(ElementCount, 0u);
  for (std::size_t input = 0u; input < InputCount; ++input) {
    result.values[input].resize(ElementCount);
    for (std::size_t index = 0u; index < ElementCount; ++index) {
      result.values[input][index] = 17u + input * 37u + index * (input + 9u);
      result.expected[index] += stage_value<InputLeafCount>(
          result.values[input][index], input * InputLeafCount + 1u);
    }
  }
  for (std::uint64_t &value : result.expected) {
    value = stage_value<StageLeafCount>(value, 1u) ^ 0x55ull;
    value = stage_value<StageLeafCount>(value, StageLeafCount + 1u) * 3ull;
    value = (stage_value<StageLeafCount>(value, 2u * StageLeafCount + 1u) ^
             0xa5ull) *
                5ull +
            7ull;
    value = (stage_value<StageLeafCount>(value, 3u * StageLeafCount + 1u) |
             0x11ull);
    value = (stage_value<StageLeafCount>(value, 4u * StageLeafCount + 1u) +
             0x33ull) *
                7ull ^
            0x5aull;
    value = (stage_value<StageLeafCount>(value, 5u * StageLeafCount + 1u) ^
             0xc3ull) *
                9ull +
            13ull;
  }
  return result;
}

} // namespace

Preparation prepare_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_program(device);
  if (!program || !validate_program(*program)) {
    std::fprintf(stderr, "Graph depth-seven compile reason=%u\n",
                 static_cast<unsigned>(program.reason()));
    return {.reason = 1};
  }
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  constexpr std::size_t LogicalBytes = ElementCount * sizeof(std::uint64_t);
  constexpr std::size_t PageBytes = FrameElements * sizeof(std::uint64_t);
  auto data = make_data();
  auto inputs = std::move(data.values);
  auto expected = std::move(data.expected);
  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> backings{};
  for (std::size_t input = 0u; input < InputCount; ++input) {
    backings[input] =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
    if (backings[input] == nullptr ||
        !backings[input]->seed(std::as_bytes(std::span{inputs[input]}))) {
      return {.reason = 2};
    }
  }
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
  auto a = virtual_buffer<std::uint64_t>(ElementCount, backings[0u]);
  auto b = virtual_buffer<std::uint64_t>(ElementCount, backings[1u]);
  auto output = virtual_buffer<std::uint64_t>(ElementCount, output_backing);
  if (!a || !b || !output) {
    return {.reason = 3};
  }
  auto prepared =
      virtual_pipeline(*program, *a, *b, *output, ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr, "Graph depth-seven prepare reason=%u\n",
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

ResidentPreparation prepare_resident_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_program(device);
  if (!program || !validate_program(*program)) {
    std::fprintf(stderr, "Graph depth-seven resident compile reason=%u\n",
                 static_cast<unsigned>(program.reason()));
    return {.reason = 1};
  }
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  auto data = make_data();
  auto values = std::move(data.values);
  auto expected = std::move(data.expected);

  std::array<std::shared_ptr<VirtualBacking>, InputCount> inputs{};
  for (std::size_t input = 0u; input < InputCount; ++input) {
    auto backing =
        resident_virtual_backing<std::uint64_t>(device, ElementCount);
    if (!backing ||
        !backing.value()->write(
            0u, std::as_bytes(std::span<const std::uint64_t>{values[input]}))) {
      return {.reason = 2};
    }
    inputs[input] = std::move(backing).value();
  }
  auto output = resident_virtual_backing<std::uint64_t>(device, ElementCount);
  if (!output) {
    return {.reason = 3};
  }
  auto first = virtual_buffer<std::uint64_t>(ElementCount, inputs[0u]);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, inputs[1u]);
  auto result = virtual_buffer<std::uint64_t>(ElementCount, output.value());
  if (!first || !second || !result) {
    return {.reason = 4};
  }
  auto prepared =
      virtual_pipeline(*program, *first, *second, *result, ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr,
                 "Graph depth-seven resident prepare reason=%u native=%s\n",
                 static_cast<unsigned>(prepared.reason()),
                 prepared.location().native_reason_key == nullptr
                     ? "none"
                     : prepared.location().native_reason_key);
    return {.reason = 5};
  }
  Pipeline pipeline = std::move(prepared).value();
  if (pipeline.plan().residency.frame_capacity != FrameCapacity) {
    return {.reason = 6};
  }
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<ResidentCase>(ResidentCase{
              .expected = std::move(expected),
              .inputs = std::move(inputs),
              .output = std::move(output).value(),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

bool run_resident_case(ResidentCase &test_case,
                       std::array<ResidentObservation, RunCount> &observations,
                       rund::compute::Device &device,
                       const rund::compute::Backend backend) {
  for (ResidentObservation &observation : observations) {
    ProductRouteObservation route{};
    {
      ProductRouteScope route_scope{device, route};
      if (!route_scope) {
        return false;
      }
      observation.before = test_case.pipeline.stats();
      observation.version_before =
          rund::compute::detail::VirtualBackingAccess::version(
              *test_case.output);
      resident_stage_generations(test_case,
                                 std::span{observation.generations_before});
      observation.status = test_case.pipeline.run();
      if (!capture_resident_run(test_case, observation)) {
        return false;
      }
    }
    ResolveProductRoute(route, backend, static_cast<bool>(observation.status));
    observation.route_kind = route.kind;
    observation.accepted_owner_mask = route.accepted_owner_mask;
    observation.accepted_owner_count = route.accepted_owner_count;
  }
  return true;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
