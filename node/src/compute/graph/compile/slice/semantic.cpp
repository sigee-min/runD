#include "semantic.hpp"

#include "../../../expression/state.hpp"
#include "../../state.hpp"

#include <rund/compute/abi/graph.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <span>
#include <variant>
#include <vector>

namespace rund::compute::detail::graph_compile {
namespace {

constexpr std::uint32_t Visiting = std::numeric_limits<std::uint32_t>::max();

struct Producer final {
  const MapStep *map{};
  std::size_t output{};
};

struct Composer final {
  const GraphState &graph;
  Type type{Type::I32};
  std::span<const std::uint32_t> interface_values{};
  std::shared_ptr<ExprState> expression{std::make_shared<ExprState>()};
  std::vector<Producer> producers{};
  std::vector<std::uint32_t> values{};
  std::vector<std::uint32_t> input_nodes{};

  [[nodiscard]] std::uint32_t append(ExprNode node) {
    if (expression->nodes.size() >= ExpressionCapacity) {
      return 0u;
    }
    expression->nodes.push_back(std::move(node));
    return static_cast<std::uint32_t>(expression->nodes.size());
  }

  [[nodiscard]] std::uint32_t interface_input(const std::size_t ordinal) {
    if (ordinal >= input_nodes.size()) {
      return 0u;
    }
    std::uint32_t &input_node = input_nodes[ordinal];
    if (input_node == 0u) {
      input_node = append(ExprNode{
          .operation = ExprOp::Input,
          .type = type,
          .fixed_format = {},
          .left = static_cast<std::uint32_t>(ordinal),
      });
    }
    return input_node;
  }

  [[nodiscard]] std::uint32_t
  clone_node(const ExprState &source, const std::uint32_t reference,
             const std::span<const std::uint32_t> routes,
             std::vector<std::uint32_t> &mapping) {
    if (reference == 0u || reference > source.nodes.size()) {
      return 0u;
    }
    std::uint32_t &stored = mapping[reference - 1u];
    if (stored == Visiting) {
      return 0u;
    }
    if (stored != 0u) {
      return stored;
    }
    stored = Visiting;
    ExprNode node = source.nodes[reference - 1u];
    if (node.operation == ExprOp::Input) {
      if (node.left >= routes.size()) {
        stored = 0u;
        return 0u;
      }
      stored = value(routes[node.left]);
      return stored;
    }
    const auto operand = [&](std::uint32_t &value) {
      value = clone_node(source, value, routes, mapping);
      return value != 0u;
    };
    const std::uint8_t arity = expr_arity(node.operation);
    if (arity == InvalidArity || (arity >= 1u && !operand(node.left)) ||
        (arity >= 2u && !operand(node.right)) ||
        (arity == 3u && !operand(node.third))) {
      stored = 0u;
      return 0u;
    }
    stored = append(node);
    return stored;
  }

  [[nodiscard]] std::uint32_t value(const std::uint32_t id) {
    if (id == 0u || id >= values.size()) {
      return 0u;
    }
    const auto interface =
        std::find(interface_values.begin(), interface_values.end(), id);
    if (interface != interface_values.end()) {
      return interface_input(
          static_cast<std::size_t>(interface - interface_values.begin()));
    }
    std::uint32_t &stored = values[id];
    if (stored == Visiting) {
      return 0u;
    }
    if (stored != 0u) {
      return stored;
    }
    const Producer producer = producers[id];
    if (producer.map == nullptr ||
        producer.output >= producer.map->expressions.size() ||
        !graph.value_ids.valid(producer.map->inputs)) {
      return 0u;
    }
    const ExprRef &root = producer.map->expressions[producer.output];
    if (root.state == nullptr || root.node == 0u ||
        root.node > root.state->nodes.size() || root.type != type ||
        root.fixed_format != FixedFormat{}) {
      return 0u;
    }
    stored = Visiting;
    std::vector<std::uint32_t> mapping(root.state->nodes.size());
    const std::span<const std::uint32_t> routes =
        graph.value_ids.view(producer.map->inputs);
    const std::uint32_t projected =
        clone_node(*root.state, root.node, routes, mapping);
    stored = projected;
    return stored;
  }
};

[[nodiscard]] bool index_producers(Composer &composer) {
  composer.producers.resize(composer.graph.values.size() + 1u);
  composer.values.resize(composer.graph.values.size() + 1u);
  for (const GraphStep &step : composer.graph.steps) {
    const auto *const map = std::get_if<MapStep>(&step);
    if (map == nullptr) {
      continue;
    }
    if (!map->reads.empty() || !map->control.empty() ||
        !composer.graph.value_ids.valid(map->outputs)) {
      return false;
    }
    const std::span<const std::uint32_t> outputs =
        composer.graph.value_ids.view(map->outputs);
    if (outputs.size() != map->expressions.size()) {
      return false;
    }
    for (std::size_t output = 0u; output < outputs.size(); ++output) {
      const std::uint32_t value = outputs[output];
      if (value == 0u || value >= composer.producers.size() ||
          composer.producers[value].map != nullptr) {
        return false;
      }
      composer.producers[value] = Producer{.map = map, .output = output};
    }
  }
  return true;
}

[[nodiscard]] bool
distinct_nonzero(const std::span<const std::uint32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (values[index] == 0u ||
        std::find(values.begin(), values.begin() + index, values[index]) !=
            values.begin() + index) {
      return false;
    }
  }
  return true;
}

} // namespace

Result<std::shared_ptr<ProgramState>> compile_service_free_map_semantic_inputs(
    const GraphState &source, const Type type, const std::size_t capacity,
    const std::span<const std::uint32_t> inputs,
    const std::uint32_t terminal_value) {
  try {
    if (!source.status || source.device == nullptr || inputs.empty() ||
        inputs.size() > MaxMapInputs || source.inputs.size() != inputs.size() ||
        !std::equal(inputs.begin(), inputs.end(), source.inputs.begin()) ||
        !distinct_nonzero(inputs) || (type != Type::U32 && type != Type::U64) ||
        capacity == 0u) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::PrimitiveUnsupported);
    }
    Composer composer{.graph = source,
                      .type = type,
                      .interface_values = inputs,
                      .input_nodes =
                          std::vector<std::uint32_t>(inputs.size(), 0u)};
    if (!index_producers(composer)) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::PrimitiveUnsupported);
    }
    const std::uint32_t root = composer.value(terminal_value);
    if (root == 0u || root > composer.expression->nodes.size()) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::ExpressionCapacity);
    }
    const auto graph =
        make_graph(source.device, "tiled-map-service-free", capacity);
    if (graph == nullptr) {
      return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
    }
    std::vector<std::uint32_t> graph_inputs;
    graph_inputs.reserve(inputs.size());
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
      const std::uint32_t input = graph_input_count(graph, type, capacity);
      if (input == 0u) {
        return Result<std::shared_ptr<ProgramState>>::fail(
            graph->status.reason());
      }
      graph_inputs.push_back(input);
    }
    const std::uint32_t count = graph_input_count(graph, Type::U64, 1u);
    if (count == 0u) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status.reason());
    }
    try {
      graph->bounded_inputs.push_back(
          BoundedInputSchema{.count = count, .capacity = capacity});
    } catch (const std::bad_alloc &) {
      return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
    }
    const std::array expressions{
        ExprRef{composer.expression, root, type, FixedFormat{}}};
    const ValueIds outputs = graph_map_multi_controlled(
        graph, graph_inputs, expressions,
        FlowControl{.count = count, .capacity = capacity},
        "service-free-map-dag", capacity);
    if (!graph->status || outputs.size() != 1u) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status ? Reason::GraphBindingInvalid : graph->status.reason());
    }
    graph_output(graph, outputs.front());
    if (!graph->status) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          graph->status.reason());
    }
    std::vector<Type> input_types(inputs.size(), type);
    input_types.push_back(Type::U64);
    const std::array output_types{type};
    return compile_graph(graph, input_types, output_types);
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramCapacity);
  } catch (const std::length_error &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramCapacity);
  }
}

} // namespace rund::compute::detail::graph_compile
