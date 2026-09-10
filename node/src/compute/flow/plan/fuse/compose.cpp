#include "internal.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail::flow_fuse {
namespace {

constexpr std::uint32_t Visiting = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] bool contains(const std::span<const std::uint32_t> values,
                            const std::uint32_t value) noexcept {
  return std::find(values.begin(), values.end(), value) != values.end();
}

[[nodiscard]] std::size_t
input_index(const std::span<const std::uint32_t> inputs,
            const std::uint32_t value) noexcept {
  const auto position = std::find(inputs.begin(), inputs.end(), value);
  return position == inputs.end()
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(position - inputs.begin());
}

struct Copy final {
  const ExprState *source{};
  bool producer{};
  std::vector<std::uint32_t> nodes{};
};

struct Composer final {
  const MapRecipe &producer;
  const MapRecipe &consumer;
  std::vector<std::uint32_t> inputs{};
  std::shared_ptr<ExprState> state{};
  std::vector<std::uint32_t> input_nodes{};
  std::vector<Copy> copies{};

  [[nodiscard]] std::uint32_t append(ExprNode node) {
    if (state->nodes.size() >= ExpressionCapacity) {
      return 0u;
    }
    state->nodes.push_back(std::move(node));
    return static_cast<std::uint32_t>(state->nodes.size());
  }

  [[nodiscard]] Copy &copy(const ExprState &source, const bool from_producer) {
    const auto found =
        std::find_if(copies.begin(), copies.end(), [&](const Copy &entry) {
          return entry.source == &source && entry.producer == from_producer;
        });
    if (found != copies.end()) {
      return *found;
    }
    copies.push_back(
        Copy{.source = &source,
             .producer = from_producer,
             .nodes = std::vector<std::uint32_t>(source.nodes.size())});
    return copies.back();
  }

  [[nodiscard]] std::uint32_t input(const ExprNode &node,
                                    const std::uint32_t value) {
    const std::size_t ordinal = input_index(inputs, value);
    if (ordinal >= inputs.size()) {
      return 0u;
    }
    std::uint32_t &stored = input_nodes[ordinal];
    if (stored != 0u) {
      const ExprNode &existing = state->nodes[stored - 1u];
      return existing.operation == ExprOp::Input &&
                     existing.type == node.type &&
                     existing.fixed_format == node.fixed_format
                 ? stored
                 : 0u;
    }
    stored = append(ExprNode{
        .operation = ExprOp::Input,
        .type = node.type,
        .fixed_format = node.fixed_format,
        .left = static_cast<std::uint32_t>(ordinal),
    });
    return stored;
  }

  [[nodiscard]] std::uint32_t clone(const ExprRef &expression,
                                    const bool from_producer) {
    if (expression.state == nullptr || expression.node == 0u ||
        expression.node > expression.state->nodes.size()) {
      return 0u;
    }
    Copy &mapping = copy(*expression.state, from_producer);
    return clone_node(*expression.state, expression.node, from_producer,
                      mapping);
  }

  [[nodiscard]] std::uint32_t clone_node(const ExprState &source,
                                         const std::uint32_t reference,
                                         const bool from_producer,
                                         Copy &mapping) {
    if (reference == 0u || reference > source.nodes.size()) {
      return 0u;
    }
    std::uint32_t &stored = mapping.nodes[reference - 1u];
    if (stored == Visiting) {
      return 0u;
    }
    if (stored != 0u) {
      return stored;
    }
    stored = Visiting;
    ExprNode node = source.nodes[reference - 1u];
    if (node.operation == ExprOp::Input) {
      const std::span<const std::uint32_t> route =
          from_producer ? std::span<const std::uint32_t>{producer.inputs}
                        : std::span<const std::uint32_t>{consumer.inputs};
      if (node.left >= route.size()) {
        stored = 0u;
        return 0u;
      }
      const std::uint32_t value = route[node.left];
      if (!from_producer) {
        const auto output =
            std::find(producer.outputs.begin(), producer.outputs.end(), value);
        if (output != producer.outputs.end()) {
          const std::size_t ordinal =
              static_cast<std::size_t>(output - producer.outputs.begin());
          if (ordinal >= producer.expressions.size()) {
            stored = 0u;
            return 0u;
          }
          stored = clone(producer.expressions[ordinal], true);
          return stored;
        }
      }
      stored = input(node, value);
      return stored;
    }

    const auto operand = [&](std::uint32_t &value) {
      value = clone_node(source, value, from_producer, mapping);
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
};

} // namespace

[[nodiscard]] Compose compose(const MapRecipe &producer, MapRecipe &consumer) {
  try {
    Composer merged{
        .producer = producer,
        .consumer = consumer,
        .state = std::make_shared<ExprState>(),
    };
    merged.inputs.reserve(producer.inputs.size() + consumer.inputs.size());
    for (const std::uint32_t value : consumer.inputs) {
      if (!contains(producer.outputs, value) &&
          !contains(merged.inputs, value)) {
        merged.inputs.push_back(value);
      }
    }
    for (const std::uint32_t value : producer.inputs) {
      if (!contains(merged.inputs, value)) {
        merged.inputs.push_back(value);
      }
    }
    if (merged.inputs.size() > MaxMapInputs) {
      return Compose::Limit;
    }
    merged.input_nodes.resize(merged.inputs.size());
    merged.copies.reserve(producer.expressions.size() +
                          consumer.expressions.size());
    merged.state->nodes.reserve(
        std::min(ExpressionCapacity,
                 producer.expressions.size() * ExpressionCapacity +
                     consumer.expressions.size() * ExpressionCapacity));

    std::vector<ExprRef> expressions;
    expressions.reserve(consumer.expressions.size());
    for (const ExprRef &expression : consumer.expressions) {
      const std::uint32_t root = merged.clone(expression, false);
      if (root == 0u) {
        return merged.state->nodes.size() >= ExpressionCapacity
                   ? Compose::Limit
                   : Compose::Invalid;
      }
      expressions.push_back(ExprRef{merged.state, root, expression.type,
                                    expression.fixed_format});
    }
    const std::size_t operation_nodes = static_cast<std::size_t>(std::count_if(
        merged.state->nodes.begin(), merged.state->nodes.end(),
        [](const ExprNode &node) { return node.operation != ExprOp::Input; }));
    // Dynamic lowering materializes one read per binding, one canonical index,
    // every non-Input expression node, and one write per root. BuildContext
    // may canonicalize this further, but this upper bound must fit before the
    // fused Map can become the sole physical authority.
    if (merged.inputs.size() + 1u + operation_nodes + expressions.size() >
        ExpressionCapacity) {
      return Compose::Limit;
    }
    consumer.inputs = std::move(merged.inputs);
    consumer.expressions = std::move(expressions);
    return Compose::Applied;
  } catch (const std::bad_alloc &) {
    throw;
  }
}

} // namespace rund::compute::detail::flow_fuse
