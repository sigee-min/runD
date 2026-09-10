#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_resident_host {
namespace {

[[nodiscard]] bool seed(const std::shared_ptr<MemoryVirtualBacking> &backing,
                        const std::span<const std::uint64_t> values) noexcept {
  return backing != nullptr && backing->seed(std::as_bytes(values));
}

} // namespace

Preparation prepare_case(const rund::compute::Device &device) {
  auto program = build_program(device);
  if (!program || !validate_program(*program)) {
    return {.reason = 1};
  }

  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> expected(ElementCount);
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    for (std::size_t input = 0u; input < InputCount; ++input) {
      values[input].push_back(Workload::input_value(input, index));
    }
    expected[index] = Workload::expected_value(index);
  }

  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> inputs{};
  for (std::size_t input = 0u; input < InputCount; ++input) {
    inputs[input] =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
    if (!seed(inputs[input], std::span{values[input]})) {
      return {.reason = 2};
    }
  }
  auto output = std::make_shared<MemoryVirtualBacking>(LogicalBytes, PageBytes);
  auto first =
      rund::compute::virtual_buffer<std::uint64_t>(ElementCount, inputs[0u]);
  auto second =
      rund::compute::virtual_buffer<std::uint64_t>(ElementCount, inputs[1u]);
  auto third =
      rund::compute::virtual_buffer<std::uint64_t>(ElementCount, inputs[2u]);
  auto result =
      rund::compute::virtual_buffer<std::uint64_t>(ElementCount, output);
  if (!first || !second || !third || !result) {
    return {.reason = 3};
  }
  auto prepared = rund::compute::virtual_pipeline(
      *program, *first, *second, *third, *result,
      rund::compute::ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr, "GraphResident Host prepare reason=%u\n",
                 static_cast<unsigned>(prepared.reason()));
    return {.reason = 4};
  }
  Pipeline pipeline = std::move(prepared).value();
  if (pipeline.plan().residency.frame_capacity != 2u) {
    return {.reason = 5};
  }
  auto state = rund::compute::detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<Case>(Case{
              .values = std::move(values),
              .expected = std::move(expected),
              .inputs = std::move(inputs),
              .output = std::move(output),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

bool run_case(Case &test_case, Observation &observation,
              rund::compute::Device &device) {
  ProductRouteObservation route{};
  {
    ProductRouteScope scope{device, route};
    if (!scope) {
      return false;
    }
    observation.before = test_case.pipeline.stats();
    observation.version_before =
        rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
    observation.status = test_case.pipeline.run();
    if (!capture_run(test_case, observation)) {
      return false;
    }
  }
  ResolveProductRoute(route, rund::compute::Backend::Cpu,
                      static_cast<bool>(observation.status));
  observation.route_kind = route.kind;
  observation.owner_mask = route.accepted_owner_mask;
  observation.owner_count = route.accepted_owner_count;
  observation.callbacks_quiet =
      !route.device_vsm_prepare_called && !route.device_vsm_execute_called &&
      !route.device_vsm_execute_accepted && !route.sliding_prepare_called &&
      !route.sliding_execute_called && !route.sliding_execute_accepted &&
      !route.submit_residency_pipeline_called &&
      !route.submit_residency_pipeline_accepted &&
      !route.submit_residency_stream_window_called &&
      !route.submit_residency_stream_window_accepted &&
      !route.submit_residency_schedule_called &&
      !route.submit_residency_schedule_accepted &&
      !route.prepare_residency_sliding_called &&
      !route.submit_residency_sliding_called &&
      !route.submit_residency_sliding_accepted && !route.window_submit_called &&
      !route.window_submit_accepted;
  return true;
}

} // namespace rund_node_test_virtual::product::graph_resident_host
