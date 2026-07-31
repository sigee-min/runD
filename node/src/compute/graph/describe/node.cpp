#include "model.hpp"

#include "../../map/build.hpp"
#include "../../type.hpp"
#include "../local.hpp"
#include "../scan.hpp"

#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/scan/identity.hpp>

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

void append_access(graph::Info &info, graph::Node &node,
                   const std::uint32_t resource,
                   const resource::AccessMode mode) {
  const graph::Resource &value = info.resources[resource - 1u];
  node.accesses.push_back(graph::Access{.resource = resource,
                                        .mode = mode,
                                        .offset_bytes = 0u,
                                        .size_bytes = value.bytes,
                                        .element_bytes = value.element_bytes,
                                        .element_count = value.elements,
                                        .stride_bytes = value.element_bytes});
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

[[nodiscard]] Status build_map(const GraphState &state, const MapStep &map,
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

[[nodiscard]] Status build_scan(const GraphState &state, const ScanStep &scan,
                                Draft &draft, graph::Node &info,
                                std::vector<kernel::GraphBufferRef> &refs,
                                kernel::GraphNode &canonical,
                                resource_detail::MemoryNode &memory) {
  refs.push_back({scan.input, kernel::BufferRole::Read});
  append_access(draft.description.info, info, scan.input,
                resource::AccessMode::Read);
  if (scan.count != 0u) {
    refs.push_back({scan.count, kernel::BufferRole::Read});
    append_access(draft.description.info, info, scan.count,
                  resource::AccessMode::Read);
  }
  refs.push_back({scan.output, kernel::BufferRole::Write});
  append_access(draft.description.info, info, scan.output,
                resource::AccessMode::Write);

  const auto operation = scan_operation(scan.operation);
  if (!operation) {
    return Status::fail(Reason::ScanOpUnsupported);
  }
  const auto element = scan_element(state.values[scan.input - 1u].type);
  if (!element) {
    return Status::fail(Reason::TypeUnsupported);
  }
  const kernel::ScanDesc desc{
      .op = *operation,
      .element = *element,
      .element_count = state.values[scan.input - 1u].count,
      .block_size = collective_block(state.values[scan.input - 1u].count),
      .count_source =
          scan.count == 0u
              ? kernel::ComputeCountSource::Descriptor
              : (type_bytes(state.values[scan.count - 1u].type) == 8u
                     ? kernel::ComputeCountSource::BufferU64
                     : kernel::ComputeCountSource::BufferU32)};
  const kernel::ScanHash hash = kernel::HashScan(desc);
  kernel::GraphControl control{};
  if (!scan.control.empty() || scan.control.iteration != 0u) {
    if (scan.count == 0u || scan.control.count != scan.count ||
        scan.control.predicate != 0u ||
        scan.control.capacity != state.values[scan.input - 1u].count ||
        scan.control.iteration == 0u) {
      return Status::fail(Reason::BoundedCountInvalid);
    }
    control = kernel::GraphControl{
        .count_source = state.values[scan.count - 1u].type == Type::U64
                            ? kernel::GraphControlSource::U64
                            : kernel::GraphControlSource::U32,
        .count_binding = 1u,
        .capacity = scan.control.capacity,
        .iteration = scan.control.iteration,
    };
    if (!control.valid(refs.size())) {
      return Status::fail(Reason::BoundedCountInvalid);
    }
  }

  info.operation = graph::Operation::Scan;
  info.elements = desc.element_count;
  canonical = kernel::GraphNode{.buffers = refs.data(),
                                .buffer_count = refs.size(),
                                .kind = kernel::NodeKind::Scan,
                                .primitive_hash_hi = hash.hi,
                                .primitive_hash_lo = hash.lo,
                                .element_count = desc.element_count,
                                .control = control};
  memory.domain = resource_detail::Domain{.count = scan.count};
  memory.write = memory.domain.empty() ? resource_detail::Write::Full
                                       : resource_detail::Write::Domain;
  return Status::success();
}

[[nodiscard]] Status build_primitive(const GraphState &state,
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

} // namespace

Status build_nodes(const GraphState &state, Draft &draft) {
  graph::Info &graph = draft.description.info;
  graph.nodes.reserve(state.steps.size());
  draft.description.map_operations.reserve(state.steps.size());
  draft.refs.reserve(state.steps.size());
  draft.nodes.reserve(state.steps.size());
  draft.memory.reserve(state.steps.size());

  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    draft.refs.emplace_back();
    std::vector<kernel::GraphBufferRef> &refs = draft.refs.back();
    graph::Node info{.index = static_cast<std::uint32_t>(index)};
    kernel::GraphNode canonical{};
    resource_detail::MemoryNode memory{};

    const GraphStep &step = state.steps[index];
    Status status = Status::success();
    if (const auto *map = std::get_if<MapStep>(&step)) {
      status = build_map(state, *map, draft, info, refs, canonical, memory);
    } else if (const auto *scan = std::get_if<ScanStep>(&step)) {
      status = build_scan(state, *scan, draft, info, refs, canonical, memory);
    } else {
      status = build_primitive(state, std::get<GraphPrimitive>(step), draft,
                               info, refs, canonical, memory);
    }
    if (!status) {
      return status;
    }
    graph.nodes.push_back(std::move(info));
    draft.nodes.push_back(canonical);
    draft.memory.push_back(memory);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_detail::describe_detail
