#include "model.hpp"

#include "../../../fixed/format.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"
#include "../../local.hpp"
#include "../model.hpp"

#include <kernel/program/compute/graph/schema.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace graph_build_detail {

bool append_primitive(GraphState &graph,
                      const std::span<const std::uint32_t> inputs,
                      const std::span<const std::uint32_t> outputs,
                      const std::uint32_t output, const Primitive primitive,
                      const PrimitiveOptions options, rund::AccelGraphNode node,
                      const FlowControl control) {
  if (inputs.empty()) {
    graph_detail::reject(graph, Reason::GraphBindingInvalid);
    return false;
  }
  try {
    const std::optional<ValueRoutes> routes =
        graph.value_ids.store(inputs, outputs);
    if (!routes) {
      graph_detail::reject(graph, Reason::GraphCapacity);
      return false;
    }
    graph.steps.emplace_back(GraphPrimitive{
        .inputs = routes->inputs,
        .outputs = routes->outputs,
        .output = output,
        .primitive = primitive,
        .options = options,
        .node = std::move(node),
        .control = control,
    });
    return true;
  } catch (const std::bad_alloc &) {
    graph_detail::reject(graph, Reason::GraphCapacity);
    return false;
  }
}

} // namespace graph_build_detail

GraphOut graph_primitive(const std::shared_ptr<GraphState> &graph,
                         const Primitive primitive,
                         const std::span<const GraphArg> inputs,
                         const PrimitiveOptions options,
                         const std::span<const GraphArg> recipe_outputs,
                         const FlowControl control) {
  if (graph == nullptr || !graph->status) {
    return {};
  }
  if (inputs.empty() || graph->steps.size() >= graph_detail::MaxSteps) {
    graph_detail::reject(*graph, inputs.empty() ? Reason::GraphBindingInvalid
                                                : Reason::GraphCapacity);
    return {};
  }
  if (const char *const reason =
          graph_build_detail::unsupported(primitive, options);
      reason != nullptr) {
    graph_detail::reject(*graph, project_reason(reason, Reason::GraphInvalid));
    return {};
  }
  if (primitive == Primitive::Reduce && inputs.size() == 2u &&
      (inputs[1u].count != 1u ||
       (inputs[1u].type != Type::U32 && inputs[1u].type != Type::U64))) {
    graph_detail::reject(*graph, Reason::BoundedCountInvalid);
    return {};
  }
  if ((primitive == Primitive::Sort || primitive == Primitive::Argsort) &&
      inputs.size() == 2u &&
      (inputs[1u].count != 1u ||
       (inputs[1u].type != Type::U32 && inputs[1u].type != Type::U64))) {
    graph_detail::reject(*graph, Reason::BoundedCountInvalid);
    return {};
  }

  const GraphArg *control_input = nullptr;
  for (const GraphArg &input : inputs) {
    if (!graph_detail::valid(*graph, input.value)) {
      graph_detail::reject(*graph, Reason::GraphValueInvalid);
      return {};
    }
    const GraphValue &value = graph->values[input.value - 1u];
    if (value.type != input.type || value.count != input.count ||
        value.fixed_format != input.fixed_format) {
      graph_detail::reject(*graph, Reason::GraphValueMismatch);
      return {};
    }
    if (control_input == nullptr && input.value == control.count) {
      control_input = &input;
    }
  }
  if (!control.empty() || control.iteration != 0u) {
    if ((primitive != Primitive::Sort && primitive != Primitive::Argsort) ||
        control.count == 0u || control.predicate != 0u ||
        control.iteration == 0u || control.capacity != inputs.front().count ||
        control_input == nullptr || control_input->count != 1u ||
        (control_input->type != Type::U32 &&
         control_input->type != Type::U64)) {
      graph_detail::reject(*graph, Reason::BoundedCountInvalid);
      return {};
    }
  }

  std::uint32_t primary_write = 0u;
  rund::AccelGraphNode node =
      graph_build_detail::make_node(primitive, inputs, options, primary_write);
  if (!node.signature.ok) {
    if (graph->count != 0u || recipe_outputs.empty() ||
        recipe_outputs.size() > MaxOutputs ||
        primary_write >= recipe_outputs.size()) {
      graph_detail::reject(
          *graph, project_reason(node.signature.reason, Reason::GraphInvalid));
      return {};
    }
    std::vector<std::uint32_t> outputs;
    std::vector<GraphArg> public_outputs;
    try {
      outputs.reserve(recipe_outputs.size());
      public_outputs.reserve(recipe_outputs.size());
      for (const GraphArg &recipe : recipe_outputs) {
        if (recipe.value != 0u || type_bytes(recipe.type) == 0u) {
          graph_detail::reject(*graph, Reason::GraphValueInvalid);
          return {};
        }
        const std::uint32_t output = graph_detail::append(
            *graph, recipe.type, recipe.count, recipe.fixed_format);
        if (output == 0u) {
          return {};
        }
        outputs.push_back(output);
        public_outputs.push_back(
            GraphArg{output, recipe.type, recipe.count, recipe.fixed_format});
      }
      std::vector<std::uint32_t> values;
      values.reserve(inputs.size());
      for (const GraphArg &input : inputs) {
        values.push_back(input.value);
      }
      const std::uint32_t output = outputs[primary_write];
      const GraphValue selected = graph->values[output - 1u];
      if (!graph_build_detail::append_primitive(*graph, values, outputs, output,
                                                primitive, options,
                                                std::move(node), control)) {
        return {};
      }
      return GraphOut{.value = output,
                      .type = selected.type,
                      .count = selected.count,
                      .fixed_format = selected.fixed_format,
                      .outputs = std::move(public_outputs)};
    } catch (const std::bad_alloc &) {
      graph_detail::reject(*graph, Reason::GraphCapacity);
      return {};
    }
  }

  if (!control.empty() || control.iteration != 0u) {
    const std::size_t count_ordinal =
        static_cast<std::size_t>(control_input - inputs.data());
    std::size_t read_ordinal = 0u;
    std::uint32_t count_binding = kernel::kNoGraphControlBinding;
    for (std::size_t port = 0u; port < node.signature.value_count; ++port) {
      if (node.signature.values[port].role != kernel::BufferRole::Read) {
        continue;
      }
      if (read_ordinal++ == count_ordinal) {
        count_binding = static_cast<std::uint32_t>(port);
        break;
      }
    }
    node.control = kernel::GraphControl{
        .count_source = control_input->type == Type::U64
                            ? kernel::GraphControlSource::U64
                            : kernel::GraphControlSource::U32,
        .count_binding = count_binding,
        .capacity = control.capacity,
        .iteration = control.iteration,
    };
    if (!node.control.valid(node.signature.value_count)) {
      graph_detail::reject(*graph, Reason::BoundedCountInvalid);
      return {};
    }
  }

  std::size_t read_count = 0u;
  std::vector<std::uint32_t> outputs;
  try {
    outputs.reserve(node.signature.value_count);
  } catch (const std::bad_alloc &) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return {};
  }
  const Type payload_type =
      primitive == Primitive::Partition && inputs.size() == 2u
          ? inputs[1u].type
          : inputs.front().type;
  FixedFormat payload_format =
      primitive == Primitive::Partition && inputs.size() == 2u
          ? inputs[1u].fixed_format
          : inputs.front().fixed_format;
  if (payload_type == Type::FixedLane32 || payload_type == Type::FixedLane64) {
    if (primitive == Primitive::Transform || primitive == Primitive::Factor ||
        primitive == Primitive::Solve || primitive == Primitive::Spectrum) {
      payload_format.approximation = Approximation::Deterministic;
    }
  }
  const bool count_nonzero = primitive == Primitive::Reduce && options.flag;
  for (std::size_t index = 0u; index < node.signature.value_count; ++index) {
    const kernel::GraphValueType &value = node.signature.values[index];
    if (value.role == kernel::BufferRole::Read) {
      ++read_count;
      continue;
    }
    Type output_type = payload_type;
    if (value.kind == kernel::GraphValueKind::Aux ||
        value.kind == kernel::GraphValueKind::Status ||
        ((primitive == Primitive::Sort || primitive == Primitive::Argsort) &&
         value.kind == kernel::GraphValueKind::Values) ||
        primitive == Primitive::Histogram || primitive == Primitive::Compact) {
      output_type = Type::U32;
    }
    if (count_nonzero) {
      output_type =
          value.element_bytes == sizeof(std::uint64_t) ? Type::U64 : Type::U32;
    }
    if (value.element_bytes != 0u &&
        value.element_bytes != type_bytes(output_type)) {
      graph_detail::reject(*graph, Reason::GraphTypeMismatch);
      return {};
    }
    const FixedFormat output_format =
        output_type == payload_type ? payload_format : FixedFormat{};
    const std::uint32_t output =
        graph_detail::append(*graph, output_type, value.count, output_format);
    if (output == 0u) {
      return {};
    }
    try {
      outputs.push_back(output);
    } catch (const std::bad_alloc &) {
      graph_detail::reject(*graph, Reason::GraphCapacity);
      return {};
    }
  }
  if (read_count != inputs.size() || primary_write >= outputs.size()) {
    graph_detail::reject(*graph, Reason::GraphBindingInvalid);
    return {};
  }

  const std::uint32_t output = outputs[primary_write];
  const GraphValue selected = graph->values[output - 1u];
  std::vector<GraphArg> public_outputs;
  try {
    public_outputs.reserve(outputs.size());
    for (const std::uint32_t value : outputs) {
      const GraphValue &entry = graph->values[value - 1u];
      public_outputs.push_back(
          GraphArg{value, entry.type, entry.count, entry.fixed_format});
    }
    std::vector<std::uint32_t> values;
    values.reserve(inputs.size());
    for (const GraphArg &input : inputs) {
      values.push_back(input.value);
    }
    if (!graph_build_detail::append_primitive(*graph, values, outputs, output,
                                              primitive, options,
                                              std::move(node), control)) {
      return {};
    }
  } catch (const std::bad_alloc &) {
    graph_detail::reject(*graph, Reason::GraphCapacity);
    return {};
  }
  return GraphOut{.value = output,
                  .type = selected.type,
                  .count = selected.count,
                  .fixed_format = selected.fixed_format,
                  .outputs = std::move(public_outputs)};
}

} // namespace rund::compute::detail
