#include <rund/compute/abi/primitive.hpp>
#include "../model.hpp"

#include <kernel/program/compute/ir.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {
namespace {

[[nodiscard]] std::optional<graph::Operation>
public_operation(const Primitive primitive) noexcept {
  switch (primitive) {
  case Primitive::SegmentedScan:
    return graph::Operation::SegmentedScan;
  case Primitive::SegmentedReduce:
    return graph::Operation::SegmentedReduce;
  case Primitive::Sort:
    return graph::Operation::Sort;
  case Primitive::Argsort:
    return graph::Operation::Argsort;
  case Primitive::Compact:
    return graph::Operation::Compact;
  case Primitive::Gather:
    return graph::Operation::Gather;
  case Primitive::Histogram:
    return graph::Operation::Histogram;
  case Primitive::Partition:
    return graph::Operation::Partition;
  case Primitive::Reduce:
    return graph::Operation::Reduce;
  case Primitive::Scatter:
    return graph::Operation::Scatter;
  case Primitive::ScatterReduce:
    return graph::Operation::ScatterReduce;
  case Primitive::Stencil:
    return graph::Operation::Stencil;
  case Primitive::Window:
    return graph::Operation::Window;
  case Primitive::Transform:
    return graph::Operation::Transform;
  case Primitive::Matrix:
    return graph::Operation::Matrix;
  case Primitive::Factor:
    return graph::Operation::Factor;
  case Primitive::Solve:
    return graph::Operation::Solve;
  case Primitive::Spectrum:
    return graph::Operation::Spectrum;
  }
  return std::nullopt;
}

[[nodiscard]] bool write_complete(const GraphPrimitive &primitive) noexcept {
  if (primitive.node.control.has_count() ||
      primitive.node.control.has_predicate()) {
    return false;
  }
  switch (primitive.primitive) {
  case Primitive::SegmentedScan:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Reduce:
  case Primitive::Stencil:
  case Primitive::Transform:
  case Primitive::Matrix:
  case Primitive::Factor:
  case Primitive::Solve:
  case Primitive::Spectrum:
    return true;
  case Primitive::Window:
    return primitive.node.window.count_source ==
           kernel::ComputeCountSource::Descriptor;
  case Primitive::Sort:
  case Primitive::Argsort:
    return primitive.node.sort.count_source ==
           kernel::ComputeCountSource::Descriptor;
  case Primitive::Gather:
    return primitive.node.gather.count_source ==
           kernel::ComputeCountSource::Descriptor;
  case Primitive::Compact:
  case Primitive::SegmentedReduce:
  case Primitive::Scatter:
    return false;
  case Primitive::ScatterReduce:
    // Backends validate every target, initialize the complete output, and only
    // then perform the ordered fold. Failed preflight cannot expose a write.
    return true;
  }
  return false;
}

[[nodiscard]] resource_detail::MemoryNode
primitive_memory(const GraphPrimitive &primitive,
                 const std::span<const std::uint32_t> inputs,
                 const std::span<const std::uint32_t> outputs,
                 const graph::Info &info) {
  resource_detail::MemoryNode memory{
      .domain =
          {
              .count = primitive.control.count,
              .predicate = primitive.control.predicate,
              .expected = primitive.control.predicate_expected,
          },
  };
  if (memory.domain.count == 0u) {
    std::size_t count_input = inputs.size();
    switch (primitive.primitive) {
    case Primitive::Sort:
    case Primitive::Argsort:
    case Primitive::Reduce:
      count_input = inputs.size() == 2u ? 1u : inputs.size();
      break;
    case Primitive::Gather:
    case Primitive::ScatterReduce:
      count_input = inputs.size() == 3u ? 2u : inputs.size();
      break;
    case Primitive::Window:
      count_input = inputs.size() == 2u ? 1u : inputs.size();
      break;
    default:
      break;
    }
    if (count_input < inputs.size()) {
      memory.domain.count = inputs[count_input];
    }
  }
  if (write_complete(primitive)) {
    memory.write = resource_detail::Write::Full;
  } else if (((primitive.primitive == Primitive::Sort ||
               primitive.primitive == Primitive::Argsort ||
               primitive.primitive == Primitive::Gather) &&
              !memory.domain.empty()) ||
             ((primitive.primitive == Primitive::Compact ||
               primitive.primitive == Primitive::Partition) &&
              !outputs.empty() &&
              info.resources[outputs.front() - 1u].active != 0u)) {
    memory.write = resource_detail::Write::Domain;
  }
  return memory;
}

} // namespace

Status build_primitive_node(const GraphState &state,
                                     const GraphPrimitive &primitive,
                                     Draft &draft, graph::Node &info,
                                     std::vector<kernel::GraphBufferRef> &refs,
                                     kernel::GraphNode &canonical,
                                     resource_detail::MemoryNode &memory) {
  const std::span<const std::uint32_t> inputs =
      state.value_ids.view(primitive.inputs);
  const std::span<const std::uint32_t> outputs =
      state.value_ids.view(primitive.outputs);
  if (inputs.empty() || outputs.empty()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  if (draft.zero_work) {
    for (const std::uint32_t resource_id : inputs) {
      refs.push_back({resource_id, kernel::BufferRole::Read});
      append_access(draft.description.info, info, resource_id,
                    resource::AccessMode::Read);
    }
    for (const std::uint32_t resource_id : outputs) {
      refs.push_back({resource_id, kernel::BufferRole::Write});
      append_access(draft.description.info, info, resource_id,
                    resource::AccessMode::Write);
    }
  } else {
    std::size_t read = 0u;
    std::size_t write = 0u;
    for (std::size_t port = 0u; port < primitive.node.signature.value_count;
         ++port) {
      const kernel::BufferRole role =
          primitive.node.signature.values[port].role;
      if ((role == kernel::BufferRole::Read && read >= inputs.size()) ||
          (role == kernel::BufferRole::Write && write >= outputs.size())) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const std::uint32_t resource_id =
          role == kernel::BufferRole::Read ? inputs[read++] : outputs[write++];
      refs.push_back({resource_id, role});
      append_access(draft.description.info, info, resource_id,
                    role == kernel::BufferRole::Read
                        ? resource::AccessMode::Read
                        : resource::AccessMode::Write);
    }
    if (read != inputs.size() || write != outputs.size()) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  }

  const auto operation = public_operation(primitive.primitive);
  if (!operation) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  info.operation = *operation;
  info.elements = primitive.node.element_count;
  switch (primitive.primitive) {
  case Primitive::Window:
    info.footprint = graph::Footprint{
        .pattern = graph::AccessPattern::Window,
        .input_elements = primitive.node.window.input_count,
        .output_elements = primitive.node.window.output_count,
        .tile_elements = primitive.node.window.output_count,
        .window_size = primitive.node.window.window_size,
        .stride = primitive.node.window.stride,
        .pad_left = primitive.node.window.pad_left,
        .operation = static_cast<std::uint32_t>(primitive.node.window.op),
        .boundary = static_cast<std::uint32_t>(primitive.node.window.boundary),
    };
    break;
  case Primitive::Reduce:
    info.footprint = graph::Footprint{
        .pattern = graph::AccessPattern::Reduction,
        .input_elements = primitive.node.reduce.element_count,
        .output_elements = 1u,
        .tile_elements = primitive.node.reduce.block_size,
        .operation = static_cast<std::uint32_t>(primitive.node.reduce.op),
    };
    break;
  default:
    info.footprint = graph::Footprint{
        .pattern = graph::AccessPattern::Indirect,
        .input_elements = primitive.node.element_count,
        .output_elements = primitive.node.element_count,
        .tile_elements = primitive.node.element_count,
    };
    break;
  }
  canonical =
      kernel::GraphNode{.buffers = refs.data(),
                        .buffer_count = refs.size(),
                        .kind = primitive.node.kind,
                        .primitive_hash_hi = primitive.node.primitive_hash_hi,
                        .primitive_hash_lo = primitive.node.primitive_hash_lo,
                        .element_count = primitive.node.element_count,
                        .control = primitive.node.control};
  memory = primitive_memory(primitive, inputs, outputs, draft.description.info);
  return Status::success();
}


} // namespace rund::compute::detail::graph_detail::describe_detail
