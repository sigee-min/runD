#include "state.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {
namespace {

template <class Mark>
[[nodiscard]] bool
plan_expression_groups(const std::span<const ExprRef> expressions,
                       const std::uint16_t live_outputs, MapLivePlan &map_plan,
                       std::vector<ExpressionGroupPlan> &plans,
                       Mark &&mark_input) {
  map_plan.expression_begin = plans.size();
  map_plan.expression_groups = 0u;
  for (std::size_t index = 0u; index < expressions.size(); ++index) {
    if ((live_outputs & live_bit(index)) == 0u) {
      continue;
    }
    const ExprRef &expression = expressions[index];
    if (expression.state == nullptr || expression.node == 0u ||
        expression.node > expression.state->nodes.size() ||
        expression.node > ExpressionCapacity) {
      return false;
    }
    std::size_t group_index = map_plan.expression_begin;
    const std::size_t group_end =
        map_plan.expression_begin + map_plan.expression_groups;
    for (; group_index < group_end; ++group_index) {
      if (plans[group_index].group == expression.state.get()) {
        break;
      }
    }
    if (group_index == group_end) {
      plans.push_back(ExpressionGroupPlan{.group = expression.state.get()});
      ++map_plan.expression_groups;
    }
    ExpressionGroupPlan &plan = plans[group_index];
    plan.outputs |= live_bit(index);
    plan.extent =
        std::max(plan.extent, static_cast<std::uint16_t>(expression.node));
    if (!plan.needed.test(expression.node - 1u)) {
      plan.needed.set(expression.node - 1u);
      ++plan.live_nodes;
    }
  }
  if (map_plan.expression_groups == 0u) {
    return false;
  }
  for (std::size_t group_index = map_plan.expression_begin;
       group_index < map_plan.expression_begin + map_plan.expression_groups;
       ++group_index) {
    ExpressionGroupPlan &plan = plans[group_index];
    for (std::size_t index = plan.extent; index-- > 0u;) {
      if (!plan.needed.test(index)) {
        continue;
      }
      const ExprNode &node = plan.group->nodes[index];
      const auto mark = [&](const std::uint32_t ref) {
        if (ref != 0u && ref <= plan.extent && !plan.needed.test(ref - 1u)) {
          plan.needed.set(ref - 1u);
          ++plan.live_nodes;
        }
      };
      if (node.operation == ExprOp::Input) {
        mark_input(node.left);
        continue;
      }
      const std::uint8_t arity = expr_arity(node.operation);
      if (arity == InvalidArity) {
        return false;
      }
      if (arity >= 1u) {
        mark(node.left);
      }
      if (arity >= 2u) {
        mark(node.right);
      }
      if (arity == 3u) {
        mark(node.third);
      }
    }
  }
  return true;
}

} // namespace

[[nodiscard]] Status
plan_liveness(const FlowState &flow, std::vector<bool> &needed,
              std::vector<bool> &keep, std::vector<MapLivePlan> &map_plans,
              std::vector<ExpressionGroupPlan> &expression_plans,
              std::size_t &live_steps) {
  const auto valid_value = [&](const std::uint32_t value) {
    return value != 0u && value <= flow.values.size();
  };
  if (flow.values.empty() || flow.inputs.empty() ||
      !std::all_of(flow.inputs.begin(), flow.inputs.end(), valid_value) ||
      (!flow.outputs.empty() &&
       !std::all_of(flow.outputs.begin(), flow.outputs.end(), valid_value)) ||
      (flow.outputs.empty() && !valid_value(flow.output)) ||
      (!flow.logical_outputs.empty() &&
       !std::all_of(flow.logical_outputs.begin(), flow.logical_outputs.end(),
                    valid_value))) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  for (std::size_t index = 0u; index < flow.values.size(); ++index) {
    const std::uint32_t guard = flow.values[index].guard;
    if (guard != 0u && guard >= index + 1u) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  }
  try {
    needed.assign(flow.values.size() + 1u, false);
    keep.assign(flow.steps.size(), false);
    map_plans.assign(flow.steps.size(), MapLivePlan{});
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  try {
    expression_plans.clear();
    expression_plans.reserve(flow.steps.size());
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::ExpressionCapacity);
  }
  live_steps = 0u;
  const auto require = [&](std::uint32_t value) {
    while (value != 0u) {
      if (!valid_value(value)) {
        return false;
      }
      if (needed[value]) {
        break;
      }
      needed[value] = true;
      value = flow.values[value - 1u].guard;
    }
    return true;
  };
  if (flow.outputs.empty()) {
    if (!require(flow.output)) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  } else {
    for (const std::uint32_t output : flow.outputs) {
      if (!require(output)) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
    }
  }
  for (std::size_t index = flow.steps.size(); index-- > 0u;) {
    const FlowStep &step = flow.steps[index];
    bool live = false;
    if (const auto *map = std::get_if<MapStep>(&step)) {
      if (!flow.value_ids.valid(map->inputs) ||
          !flow.value_ids.valid(map->outputs)) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const std::span<const std::uint32_t> map_inputs =
          flow.value_ids.view(map->inputs);
      const std::span<const std::uint32_t> map_outputs =
          flow.value_ids.view(map->outputs);
      if (map_inputs.size() > MaxMapInputs || map_outputs.empty() ||
          map_outputs.size() != map->expressions.size() ||
          !std::all_of(map_inputs.begin(), map_inputs.end(), valid_value) ||
          !std::all_of(map_outputs.begin(), map_outputs.end(), valid_value) ||
          (map->control.count != 0u && !valid_value(map->control.count)) ||
          (map->control.predicate != 0u &&
           !valid_value(map->control.predicate))) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      live = std::any_of(
          map_outputs.begin(), map_outputs.end(),
          [&](const std::uint32_t output) { return needed[output]; });
      if (live) {
        MapLivePlan &map_plan = map_plans[index];
        for (std::size_t output = 0u; output < map_outputs.size(); ++output) {
          if (!needed[map_outputs[output]]) {
            continue;
          }
          map_plan.live_outputs |= live_bit(output);
        }
        bool invalid_input = false;
        bool expressions_planned = false;
        try {
          expressions_planned = plan_expression_groups(
              map->expressions, map_plan.live_outputs, map_plan,
              expression_plans, [&](const std::uint32_t input) {
                if (input < map_inputs.size()) {
                  map_plan.used_inputs |= live_bit(input);
                } else {
                  invalid_input = true;
                }
              });
        } catch (const std::bad_alloc &) {
          return Status::fail(Reason::ExpressionCapacity);
        }
        if (!expressions_planned) {
          return Status::fail(Reason::ExpressionInvalid);
        }
        if (invalid_input) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
        for (std::size_t input = 0u; input < map_inputs.size(); ++input) {
          if ((map_plan.used_inputs & live_bit(input)) != 0u &&
              !require(map_inputs[input])) {
            return Status::fail(Reason::GraphBindingInvalid);
          }
        }
        if ((map->control.count != 0u && !require(map->control.count)) ||
            (map->control.predicate != 0u &&
             !require(map->control.predicate))) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
      }
    } else if (const auto *scan = std::get_if<ScanStep>(&step)) {
      if (!valid_value(scan->input) || !valid_value(scan->output) ||
          (scan->count != 0u && !valid_value(scan->count))) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      live = needed[scan->output];
      if (live) {
        if (!require(scan->input) ||
            (scan->count != 0u && !require(scan->count))) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
      }
    } else {
      const auto &primitive = std::get<FlowPrimitive>(step);
      const std::span<const std::uint32_t> primitive_inputs =
          flow.value_ids.view(primitive.inputs);
      const std::span<const std::uint32_t> primitive_outputs =
          flow.value_ids.view(primitive.outputs);
      if (primitive_inputs.empty() || primitive_outputs.empty() ||
          !std::all_of(primitive_inputs.begin(), primitive_inputs.end(),
                       valid_value) ||
          !std::all_of(primitive_outputs.begin(), primitive_outputs.end(),
                       valid_value)) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      live = std::any_of(
          primitive_outputs.begin(), primitive_outputs.end(),
          [&](const std::uint32_t output) { return needed[output]; });
      if (live) {
        for (const std::uint32_t input : primitive_inputs) {
          if (!require(input)) {
            return Status::fail(Reason::GraphBindingInvalid);
          }
        }
      }
    }
    keep[index] = live;
    live_steps += static_cast<std::size_t>(live);
  }
  return Status::success();
}

} // namespace rund::compute::detail
