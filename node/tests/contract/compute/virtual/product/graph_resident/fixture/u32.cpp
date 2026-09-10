#include "../internal.hpp"

#include "src/compute/backend/accel/diagnostic.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_resident {

U32Preparation prepare_u32_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_u32_program(device);
  if (!program || !validate_u32_program(*program)) {
    return {.reason = 21};
  }

  std::array<std::vector<std::uint32_t>, InputCount> values{};
  std::vector<std::uint32_t> expected(ElementCount);
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    for (std::size_t input = 0u; input < InputCount; ++input) {
      values[input].push_back(U32Workload::input_value(input, index));
    }
    expected[index] = U32Workload::expected_value(index);
  }

  std::array<std::shared_ptr<VirtualBacking>, InputCount> inputs{};
  for (std::size_t input = 0u; input < InputCount; ++input) {
    auto backing =
        resident_virtual_backing<std::uint32_t>(device, ElementCount);
    if (!backing ||
        !seed_backing(*backing, std::as_bytes(std::span{values[input]}))) {
      return {.reason = 22};
    }
    inputs[input] = std::move(backing).value();
  }
  auto output = resident_virtual_backing<std::uint32_t>(device, ElementCount);
  if (!output) {
    return {.reason = 23};
  }

  auto first = virtual_buffer<std::uint32_t>(ElementCount, inputs[0u]);
  auto second = virtual_buffer<std::uint32_t>(ElementCount, inputs[1u]);
  auto third = virtual_buffer<std::uint32_t>(ElementCount, inputs[2u]);
  auto result = virtual_buffer<std::uint32_t>(ElementCount, output.value());
  if (!first || !second || !third || !result) {
    return {.reason = 24};
  }
  rund::node::accel::diagnostic::ScopedAccelCompileDiagnostic diagnostics;
  auto prepared = virtual_pipeline(*program, *first, *second, *third, *result,
                                   ResidencyConfig{});
  if (!prepared) {
    std::fprintf(stderr, "GraphResident U32 prepare reason=%u native=%s\n",
                 static_cast<unsigned>(prepared.reason()),
                 prepared.location().native_reason_key == nullptr
                     ? "none"
                     : prepared.location().native_reason_key);
    const auto &snapshot = diagnostics.snapshot();
    std::fprintf(stderr, "GraphResident U32 diagnostics count=%u overflow=%u\n",
                 snapshot.count, snapshot.overflow ? 1u : 0u);
    for (std::size_t index = 0u; index < snapshot.count; ++index) {
      const auto &event = snapshot.events[index];
      std::fprintf(
          stderr,
          " GraphResident U32 diagnostic seq=%llu phase=%u node_count=%llu "
          "tiles=%llu binds=%llu mode=%u failed_node=%u raw=%u inner=%.*s "
          "inner_code=%llu projected=%.*s projected_code=%llu\n",
          static_cast<unsigned long long>(event.sequence),
          static_cast<unsigned>(event.phase),
          static_cast<unsigned long long>(event.node_count),
          static_cast<unsigned long long>(event.tile_count),
          static_cast<unsigned long long>(event.binding_count), event.mode,
          event.failed_node, event.raw_reason_present ? 1u : 0u,
          static_cast<int>(event.inner_reason_length),
          event.inner_reason == nullptr ? "" : event.inner_reason,
          static_cast<unsigned long long>(event.inner_reason_code),
          static_cast<int>(event.projected_reason_length),
          event.projected_reason == nullptr ? "" : event.projected_reason,
          static_cast<unsigned long long>(event.projected_reason_code));
    }
    return {.reason = 25};
  }
  if (prepared.value().plan().residency.frame_capacity !=
      Workload::FrameCapacity) {
    return {.reason = 26};
  }
  U32Pipeline pipeline = std::move(prepared).value();
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<U32Case>(U32Case{
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
