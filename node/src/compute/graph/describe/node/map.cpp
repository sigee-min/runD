#include "../model.hpp"

#include "../../../map/build.hpp"
#include "../../../type.hpp"
#include "../../local.hpp"

#include <kernel/program/compute/ir.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {

Status build_map_node(const GraphState &state, const MapStep &map,
                               Draft &draft, graph::Node &info,
                               std::vector<kernel::GraphBufferRef> &refs,
                               kernel::GraphNode &canonical,
                               resource_detail::MemoryNode &memory) {
  if (!state.value_ids.valid(map.inputs) ||
      !state.value_ids.valid(map.outputs)) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const std::span<const std::uint32_t> inputs =
      state.value_ids.view(map.inputs);
  const std::span<const std::uint32_t> outputs =
      state.value_ids.view(map.outputs);
  if (outputs.empty()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  if (!map.reads.empty() && map.reads.size() > inputs.size()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }

  std::vector<Type> input_types;
  std::vector<Type> output_types;
  input_types.reserve(inputs.size());
  output_types.reserve(outputs.size());
  for (const std::uint32_t input : inputs) {
    input_types.push_back(state.values[input - 1u].type);
    refs.push_back({input, kernel::BufferRole::Read});
    append_access(draft.description.info, info, input,
                  resource::AccessMode::Read);
  }
  for (const std::uint32_t output : outputs) {
    output_types.push_back(state.values[output - 1u].type);
    refs.push_back({output, kernel::BufferRole::Write});
    append_access(draft.description.info, info, output,
                  resource::AccessMode::Write);
  }

  kernel::GraphControl control{};
  control.capacity = map.control.capacity;
  control.predicate_expected = map.control.predicate_expected;
  control.iteration = map.control.iteration;
  const auto source = [&](const std::uint32_t value) {
    return state.values[value - 1u].type == Type::U64
               ? kernel::GraphControlSource::U64
               : kernel::GraphControlSource::U32;
  };
  if (map.control.count != 0u) {
    control.count_source = source(map.control.count);
    control.count_binding = static_cast<std::uint32_t>(refs.size());
    refs.push_back({map.control.count, kernel::BufferRole::Read});
    append_access(draft.description.info, info, map.control.count,
                  resource::AccessMode::Read);
  }
  if (map.control.predicate != 0u) {
    control.predicate_source = source(map.control.predicate);
    control.predicate_binding = static_cast<std::uint32_t>(refs.size());
    refs.push_back({map.control.predicate, kernel::BufferRole::Read});
    append_access(draft.description.info, info, map.control.predicate,
                  resource::AccessMode::Read);
  }

  const std::size_t count = state.values[outputs.front() - 1u].count;
  auto operation = build_map_operation_multi(count, output_types, input_types,
                                             map.expressions, map.reads);
  if (!operation) {
    return Status::fail(operation.reason());
  }
  draft.description.map_operations.push_back(std::move(operation).value());
  const kernel::ComputeIR &ir = draft.description.map_operations.back().ir();
  info.operation = graph::Operation::Map;
  info.elements = count;
  info.footprint = graph::Footprint{
      .pattern = graph::AccessPattern::Pointwise,
      .input_elements = count,
      .output_elements = count,
      .tile_elements = count,
  };
  canonical = kernel::GraphNode{.op_hash_hi = ir.op_hash_hi,
                                .op_hash_lo = ir.op_hash_lo,
                                .buffers = refs.data(),
                                .buffer_count = refs.size(),
                                .kind = kernel::NodeKind::Map,
                                .element_count = count,
                                .control = control};
  memory.domain = resource_detail::Domain{
      .count = map.control.count,
      .predicate = map.control.predicate,
      .expected = map.control.predicate_expected,
  };
  memory.write = memory.domain.empty() ? resource_detail::Write::Full
                                       : resource_detail::Write::Domain;
  memory.inplace =
      std::none_of(map.reads.begin(), map.reads.end(),
                   [](const MapRead read) { return read.indexed(); })
          ? resource_detail::Inplace::Pointwise
          : resource_detail::Inplace::None;
  return Status::success();
}


} // namespace rund::compute::detail::graph_detail::describe_detail
