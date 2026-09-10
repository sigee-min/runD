#include "fuse/internal.hpp"
#include "state.hpp"

#include "../../type.hpp"

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
    const bool fixed = type_fixed(type);
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

} // namespace

Status plan_maps(const FlowState &flow, const std::vector<bool> &keep,
                 const std::vector<StepLivePlan> &map_plans,
                 const std::vector<ExpressionGroupPlan> &expression_plans,
                 const std::span<const std::size_t> order,
                 std::vector<MapRecipe> &recipes,
                 std::vector<MapRecipe> &baseline,
                 std::vector<std::uint8_t> &skipped) {
  try {
    const Status built = flow_fuse::build_recipes(flow, keep, map_plans,
                                                  expression_plans, recipes);
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
      } else if (const auto *filter = std::get_if<FilterStep>(&step)) {
        const auto live = map_plans[step_index].live_outputs;
        if ((live & live_bit(0u)) != 0u) {
          use(filter->input, step_index);
          use(filter->rejected, step_index);
          producers[filter->values] = step_index;
        }
        if ((live & live_bit(1u)) != 0u) {
          use(filter->selected, step_index);
          producers[filter->count] = step_index;
        }
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
      // A public input with multiple live Map consumers is a physical
      // residency fan-out, not an internal value of one linear prefix.
      // Collapsing either branch into the join would erase the input's later
      // next-use/pin edge before the Graph residency planner can seal it.
      const bool external_fanout = std::any_of(
          producer.inputs.begin(), producer.inputs.end(),
          [&producers, &users](const std::uint32_t input) {
            return input < producers.size() && input < users.size() &&
                   producers[input] == Interface && users[input].size() > 1u;
          });
      if (external_fanout) {
        continue;
      }
      const flow_fuse::Compose composed =
          flow_fuse::compose(producer, consumer);
      if (composed == flow_fuse::Compose::Invalid) {
        return Status::fail(Reason::ExpressionInvalid);
      }
      if (composed == flow_fuse::Compose::Limit) {
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
