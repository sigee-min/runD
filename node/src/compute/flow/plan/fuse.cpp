#include "state.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

constexpr std::size_t Interface = std::numeric_limits<std::size_t>::max();
constexpr std::uint32_t Visiting = std::numeric_limits<std::uint32_t>::max();

enum class Compose : unsigned char {
  Applied,
  Limit,
  Invalid,
};

[[nodiscard]] constexpr bool same_control(const FlowControl left,
                                          const FlowControl right) noexcept {
  return std::tie(left.count, left.predicate, left.capacity,
                  left.predicate_expected, left.iteration) ==
         std::tie(right.count, right.predicate, right.capacity,
                  right.predicate_expected, right.iteration);
}

[[nodiscard]] bool same_domain(const FlowState &flow, const MapRecipe &producer,
                               const MapRecipe &consumer) noexcept {
  // Elementwise composition may substitute the producer expression only when
  // both Maps evaluate the same logical ordinals. A count-one producer feeding
  // a uniform read is a value dependency, not an elementwise fusion edge.
  if (producer.outputs.empty() || consumer.outputs.empty() ||
      producer.outputs.front() == 0u || consumer.outputs.front() == 0u ||
      producer.outputs.front() > flow.values.size() ||
      consumer.outputs.front() > flow.values.size() ||
      flow.values[producer.outputs.front() - 1u].count !=
          flow.values[consumer.outputs.front() - 1u].count) {
    return false;
  }
  Type type{Type::I32};
  FixedFormat format{};
  bool selected = false;
  const auto admit = [&](const std::uint32_t value) {
    if (value == 0u || value > flow.values.size()) {
      return false;
    }
    const FlowValue &candidate = flow.values[value - 1u];
    if (!selected) {
      type = candidate.type;
      format = candidate.fixed_format;
      selected = true;
      return true;
    }
    const bool fixed = type == Type::FixedLane32 || type == Type::FixedLane64;
    return candidate.type == type &&
           (!fixed || candidate.fixed_format == format);
  };
  return std::all_of(producer.inputs.begin(), producer.inputs.end(), admit) &&
         std::all_of(producer.outputs.begin(), producer.outputs.end(), admit) &&
         std::all_of(consumer.inputs.begin(), consumer.inputs.end(), admit) &&
         std::all_of(consumer.outputs.begin(), consumer.outputs.end(), admit);
}

void add_user(std::vector<std::size_t> &users, const std::size_t step) {
  const auto position = std::lower_bound(users.begin(), users.end(), step);
  if (position == users.end() || *position != step) {
    users.insert(position, step);
  }
}

void replace_user(std::vector<std::size_t> &users, const std::size_t from,
                  const std::size_t to) {
  const auto position = std::lower_bound(users.begin(), users.end(), from);
  if (position != users.end() && *position == from) {
    users.erase(position);
  }
  add_user(users, to);
}

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

[[nodiscard]] Status
build_recipes(const FlowState &flow, const std::vector<bool> &keep,
              const std::vector<MapLivePlan> &map_plans,
              const std::vector<ExpressionGroupPlan> &expression_plans,
              std::vector<MapRecipe> &recipes) {
  recipes.assign(flow.steps.size(), MapRecipe{});
  for (std::size_t step_index = 0u; step_index < flow.steps.size();
       ++step_index) {
    if (!keep[step_index]) {
      continue;
    }
    const auto *const map = std::get_if<MapStep>(&flow.steps[step_index]);
    if (map == nullptr) {
      continue;
    }
    const std::span<const std::uint32_t> map_inputs =
        flow.value_ids.view(map->inputs);
    const std::span<const std::uint32_t> map_outputs =
        flow.value_ids.view(map->outputs);
    const MapLivePlan &plan = map_plans[step_index];
    MapRecipe &recipe = recipes[step_index];
    recipe.name = map->name;
    recipe.control = map->control;
    std::array<std::uint32_t, MaxMapInputs> input_map{};
    input_map.fill(std::numeric_limits<std::uint32_t>::max());
    recipe.inputs.reserve(map_inputs.size());
    recipe.outputs.reserve(map_outputs.size());
    for (std::size_t input = 0u; input < map_inputs.size(); ++input) {
      if ((plan.used_inputs & live_bit(input)) == 0u) {
        continue;
      }
      input_map[input] = static_cast<std::uint32_t>(recipe.inputs.size());
      recipe.inputs.push_back(map_inputs[input]);
    }
    for (std::size_t output = 0u; output < map_outputs.size(); ++output) {
      if ((plan.live_outputs & live_bit(output)) != 0u) {
        recipe.outputs.push_back(map_outputs[output]);
      }
    }
    if (recipe.outputs.empty() ||
        !project_expressions(map->expressions, plan, expression_plans,
                             input_map, recipe.expressions) ||
        recipe.outputs.size() != recipe.expressions.size()) {
      return Status::fail(Reason::ExpressionCapacity);
    }
  }
  return Status::success();
}

} // namespace

Status plan_maps(const FlowState &flow, const std::vector<bool> &keep,
                 const std::vector<MapLivePlan> &map_plans,
                 const std::vector<ExpressionGroupPlan> &expression_plans,
                 const std::span<const std::size_t> order,
                 std::vector<MapRecipe> &recipes,
                 std::vector<MapRecipe> &baseline,
                 std::vector<std::uint8_t> &skipped) {
  try {
    const Status built =
        build_recipes(flow, keep, map_plans, expression_plans, recipes);
    if (!built) {
      return built;
    }
    baseline = recipes;
    skipped.assign(flow.steps.size(), std::uint8_t{0u});

    std::vector<std::vector<std::size_t>> users(flow.values.size() + 1u);
    std::vector<std::size_t> producers(flow.values.size() + 1u, Interface);
    const auto use = [&](const std::uint32_t value, const std::size_t step) {
      if (value != 0u && value < users.size()) {
        add_user(users[value], step);
      }
    };
    for (const std::size_t step_index : order) {
      const FlowStep &step = flow.steps[step_index];
      if (const auto *const map = std::get_if<MapStep>(&step)) {
        const MapRecipe &recipe = recipes[step_index];
        for (const std::uint32_t input : recipe.inputs) {
          use(input, step_index);
        }
        use(map->control.count, step_index);
        use(map->control.predicate, step_index);
        for (const std::uint32_t output : recipe.outputs) {
          producers[output] = step_index;
        }
      } else if (const auto *const scan = std::get_if<ScanStep>(&step)) {
        use(scan->input, step_index);
        use(scan->count, step_index);
        use(scan->control.count, step_index);
        use(scan->control.predicate, step_index);
        producers[scan->output] = step_index;
      } else {
        const FlowPrimitive &primitive = std::get<FlowPrimitive>(step);
        for (const std::uint32_t input :
             flow.value_ids.view(primitive.inputs)) {
          use(input, step_index);
        }
        use(primitive.control.count, step_index);
        use(primitive.control.predicate, step_index);
        for (const std::uint32_t output :
             flow.value_ids.view(primitive.outputs)) {
          producers[output] = step_index;
        }
      }
    }

    if (flow.outputs.empty()) {
      use(flow.output, Interface);
    } else {
      for (const std::uint32_t output : flow.outputs) {
        use(output, Interface);
      }
    }
    for (const std::uint32_t output : flow.logical_outputs) {
      use(output, Interface);
    }
    for (const FlowValue &value : flow.values) {
      use(value.guard, Interface);
      use(value.active, Interface);
      use(value.parent, Interface);
    }

    for (auto position = order.rbegin(); position != order.rend(); ++position) {
      const std::size_t producer_index = *position;
      MapRecipe &producer = recipes[producer_index];
      if (!producer.active() || producer.fused) {
        continue;
      }
      std::size_t consumer_index = Interface;
      bool eligible = true;
      for (const std::uint32_t output : producer.outputs) {
        if (output >= users.size() || users[output].size() != 1u) {
          eligible = false;
          break;
        }
        const std::size_t current = users[output].front();
        if (current == Interface ||
            (consumer_index != Interface && consumer_index != current)) {
          eligible = false;
          break;
        }
        consumer_index = current;
      }
      if (!eligible || consumer_index >= recipes.size()) {
        continue;
      }
      MapRecipe &consumer = recipes[consumer_index];
      if (!consumer.active() || consumer.fused ||
          !same_control(producer.control, consumer.control) ||
          !same_domain(flow, producer, consumer)) {
        continue;
      }
      const Compose composed = compose(producer, consumer);
      if (composed == Compose::Invalid) {
        return Status::fail(Reason::ExpressionInvalid);
      }
      if (composed == Compose::Limit) {
        continue;
      }
      producer.fused = true;
      for (const std::uint32_t input : producer.inputs) {
        if (input < users.size()) {
          replace_user(users[input], producer_index, consumer_index);
        }
      }
    }

    for (const std::size_t consumer_index : order) {
      MapRecipe &consumer = recipes[consumer_index];
      if (!consumer.active() || consumer.fused ||
          consumer.control.predicate != 0u) {
        continue;
      }
      consumer.indices.assign(consumer.inputs.size(), 0u);
      for (std::size_t input_index = 0u; input_index < consumer.inputs.size();
           ++input_index) {
        const std::uint32_t gathered = consumer.inputs[input_index];
        if (gathered >= producers.size()) {
          continue;
        }
        const std::size_t producer_index = producers[gathered];
        if (producer_index >= flow.steps.size() ||
            users[gathered].size() != 1u ||
            users[gathered].front() != consumer_index) {
          continue;
        }
        const auto *const primitive =
            std::get_if<FlowPrimitive>(&flow.steps[producer_index]);
        if (primitive == nullptr || primitive->operation != Primitive::Gather ||
            !primitive->control.empty()) {
          continue;
        }
        const std::span<const std::uint32_t> inputs =
            flow.value_ids.view(primitive->inputs);
        const std::span<const std::uint32_t> outputs =
            flow.value_ids.view(primitive->outputs);
        if ((inputs.size() != 2u && inputs.size() != 3u) ||
            outputs.size() != 1u || outputs.front() != gathered) {
          continue;
        }
        const FlowValue &source = flow.values[inputs[0u] - 1u];
        const FlowValue &index = flow.values[inputs[1u] - 1u];
        const FlowValue &output = flow.values[gathered - 1u];
        bool bounded = inputs.size() == 3u;
        bool bounded_outputs = true;
        if (bounded) {
          bounded_outputs =
              output.active == inputs[2u] &&
              std::all_of(consumer.outputs.begin(), consumer.outputs.end(),
                          [&](const std::uint32_t value) {
                            return flow.values[value - 1u].active == inputs[2u];
                          });
        }
        if (source.type != output.type ||
            source.fixed_format != output.fixed_format ||
            index.type != Type::U32 || index.count != output.count ||
            source.count == 0u ||
            source.count > std::numeric_limits<std::uint32_t>::max() ||
            output.count > std::numeric_limits<std::uint32_t>::max() ||
            !bounded_outputs || (!bounded && consumer.control.count != 0u) ||
            (bounded && consumer.control.count != 0u &&
             consumer.control.count != inputs[2u])) {
          continue;
        }
        if (bounded && consumer.control.count == 0u) {
          consumer.control.count = inputs[2u];
          consumer.control.capacity = output.count;
        }
        consumer.inputs[input_index] = inputs[0u];
        consumer.indices[input_index] = inputs[1u];
        skipped[producer_index] = std::uint8_t{1u};
      }
      if (std::none_of(consumer.indices.begin(), consumer.indices.end(),
                       [](const std::uint32_t index) { return index != 0u; })) {
        consumer.indices.clear();
      }
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail
