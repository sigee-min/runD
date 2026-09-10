#include "internal.hpp"

namespace rund_node_test_virtual::product::graph_resident {

rund::compute::Result<Program>
build_program(const rund::compute::Device &device, const Variant variant) {
  return Workload::build(device, variant);
}

bool validate_program(const Program &program, const Variant variant) noexcept {
  return Workload::validate(program, variant);
}

rund::compute::Result<U32Program>
build_u32_program(const rund::compute::Device &device) {
  return rund::compute::on(device)
      .input<std::uint32_t>(FrameElements)
      .zip_input<std::uint32_t>(FrameElements)
      .zip_input<std::uint32_t>(FrameElements)
      .branch([](auto a, auto b, auto c) {
        const auto x = a.map("graph-resident-x", [](auto value) {
          return U32Workload::add_stage<1u, LeafCount>(value);
        });
        const auto y = b.map("graph-resident-y", [](auto value) {
          return U32Workload::add_stage<
              static_cast<std::uint32_t>(LeafCount + 1u), LeafCount>(value);
        });
        const auto z =
            zip(x, y).map("graph-resident-z", [](auto left, auto right) {
              return U32Workload::add_stage<
                         static_cast<std::uint32_t>(2u * LeafCount + 1u),
                         LeafCount>(left) +
                     U32Workload::add_stage<
                         static_cast<std::uint32_t>(3u * LeafCount + 1u),
                         LeafCount>(right);
            });
        const auto w = c.map("graph-resident-w", [](auto value) {
          return U32Workload::add_stage<
              static_cast<std::uint32_t>(4u * LeafCount + 1u), LeafCount>(
              value);
        });
        return zip(z, w).map("graph-resident-output", [](auto left, auto right) {
          return U32Workload::add_stage<
                     static_cast<std::uint32_t>(5u * LeafCount + 1u),
                     LeafCount>(left) +
                 U32Workload::add_stage<
                     static_cast<std::uint32_t>(6u * LeafCount + 1u),
                     LeafCount>(right);
        });
      })
      .compile();
}

bool validate_u32_program(const U32Program &program) noexcept {
  using namespace rund::compute;
  const auto state = ::rund::compute::detail::FlowAccess::state(program);
  const auto slices = ::rund::compute::detail::graph_compile::
      compile_tiled_graph_pointwise_slices(state);
  const auto fused =
      ::rund::compute::detail::graph_compile::compile_service_free_map_program(
          state);
  if (!slices || fused || fused.reason() != Reason::ExpressionCapacity ||
      slices->resources.size() != Workload::ResourceCount ||
      slices->stages.size() != StageCount ||
      slices->input_resources.size() != InputCount ||
      slices->output_resource != Workload::ResourceCount) {
    return false;
  }
  for (std::size_t index = 0u; index < InputCount; ++index) {
    if (slices->input_resources[index] != index + 1u) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < Workload::ResourceCount; ++index) {
    const auto &resource = slices->resources[index];
    const auto expected_kind =
        Workload::ResourceRoles[index] == 0u
            ? ::rund::compute::detail::graph_compile::
                  SliceResourceKind::ExternalInput
        : Workload::ResourceRoles[index] == 2u
            ? ::rund::compute::detail::graph_compile::
                  SliceResourceKind::ExternalOutput
            : ::rund::compute::detail::graph_compile::SliceResourceKind::Internal;
    if (resource.resource != index + 1u ||
        resource.type != ::rund::compute::detail::Type::U32 ||
        resource.count != FrameElements || resource.kind != expected_kind) {
      return false;
    }
  }
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const auto &slice = slices->stages[stage];
    if (slice.inputs.size() != Workload::StageInputPorts[stage] ||
        slice.outputs.size() != Workload::StageOutputPorts[stage]) {
      return false;
    }
    const std::size_t edge_start = stage == 0u   ? 0u
                                   : stage == 1u ? 1u
                                   : stage == 2u ? 2u
                                   : stage == 3u ? 4u
                                                 : 5u;
    const std::size_t output_edge = stage == 3u   ? 4u
                                    : stage == 4u ? 5u
                                                  : stage;
    for (std::size_t input = 0u; input < slice.inputs.size(); ++input) {
      if (slice.inputs[input] != Workload::StageInputs[stage][input] + 1u ||
          slice.inputs[input] != Workload::Edges[edge_start + input][0u] +
                                      1u) {
        return false;
      }
    }
    if (slice.outputs[0u] != Workload::StageOutputs[stage] + 1u ||
        slice.outputs[0u] != Workload::Edges[output_edge][1u] + 1u) {
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_virtual::product::graph_resident
