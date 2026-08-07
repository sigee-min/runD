#include "recipe.hpp"

#include <rund/compute/abi/expression.hpp>

#include "../expression/state.hpp"
#include "../type.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute::detail {

std::shared_ptr<FlowState> make_flow(const Target target, const Type type,
                                     const std::size_t count,
                                     const FixedFormat fixed_format) {
  try {
    auto flow = std::make_shared<FlowState>();
    flow->target = target;
    flow->values.push_back(
        FlowValue{.type = type, .fixed_format = fixed_format, .count = count});
    flow->inputs.push_back(1u);
    flow->output = 1u;
    return flow;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

std::shared_ptr<FlowState>
make_flow_on(std::shared_ptr<DeviceState> device, const Type type,
             const std::size_t count, std::shared_ptr<ProgramCacheState> cache,
             const FixedFormat fixed_format) {
  if (device == nullptr) {
    return {};
  }
  auto flow = make_flow(Target::cpu(), type, count, fixed_format);
  if (flow != nullptr) {
    flow->device = std::move(device);
    flow->cache = std::move(cache);
  }
  return flow;
}

void flow_bind(const std::shared_ptr<FlowState> &flow, const HostView input) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const FlowValue &expected = flow->values[flow->inputs.front() - 1u];
  if (input.type != expected.type || input.count != expected.count ||
      (input.data == nullptr && input.count != 0u)) {
    reject(*flow, Reason::ShapeMismatch);
    return;
  }
  try {
    flow->bindings.push_back(input);
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
  }
}

std::uint32_t flow_map_value(const std::shared_ptr<FlowState> &flow,
                             const std::span<const std::uint32_t> inputs,
                             const std::string_view name, ExprRef expression) {
  const std::array expressions{expression};
  const ValueIds outputs = flow_map_multi(flow, inputs, name, expressions);
  return outputs.size() == 1u ? outputs.front() : 0u;
}

std::uint32_t
flow_map_value_controlled(const std::shared_ptr<FlowState> &flow,
                          const std::span<const std::uint32_t> inputs,
                          const std::string_view name, ExprRef expression,
                          const FlowControl control) {
  const std::array expressions{expression};
  const ValueIds outputs =
      flow_map_multi_controlled(flow, inputs, name, expressions, control);
  return outputs.size() == 1u ? outputs.front() : 0u;
}

ValueIds flow_map_multi(const std::shared_ptr<FlowState> &flow,
                        const std::span<const std::uint32_t> inputs,
                        const std::string_view name,
                        const std::span<const ExprRef> expressions) {
  return flow_map_multi_controlled(flow, inputs, name, expressions, {});
}

ValueIds flow_map_multi_controlled(const std::shared_ptr<FlowState> &flow,
                                   const std::span<const std::uint32_t> inputs,
                                   const std::string_view name,
                                   const std::span<const ExprRef> expressions,
                                   const FlowControl control) {
  if (flow == nullptr || !flow->status || inputs.empty() ||
      inputs.size() > MaxMapInputs || expressions.empty() ||
      expressions.size() > MaxOutputs) {
    return {};
  }
  if (name.empty()) {
    reject(*flow, Reason::NameEmpty);
    return {};
  }
  if (inputs.front() == 0u || inputs.front() > flow->values.size()) {
    reject(*flow, Reason::GraphValueInvalid);
    return {};
  }
  const FlowValue input = flow->values[inputs.front() - 1u];
  const auto valid_control_value = [&](const std::uint32_t value) {
    return value != 0u && value <= flow->values.size() &&
           flow->values[value - 1u].count == 1u &&
           (flow->values[value - 1u].type == Type::U32 ||
            flow->values[value - 1u].type == Type::U64);
  };
  if ((control.count != 0u &&
       (!valid_control_value(control.count) || control.capacity == 0u ||
        control.capacity != input.count)) ||
      (control.count == 0u && control.capacity != 0u) ||
      (control.predicate != 0u &&
       (!valid_control_value(control.predicate) ||
        (flow->values[control.predicate - 1u].type == Type::U32 &&
         control.predicate_expected >
             std::numeric_limits<std::uint32_t>::max()))) ||
      (control.predicate == 0u &&
       (control.predicate_expected != 0u ||
        (control.iteration != 0u && control.count == 0u)))) {
    reject(*flow, Reason::BoundedCountInvalid);
    return {};
  }
  for (const std::uint32_t value : inputs) {
    if (value == 0u || value > flow->values.size() ||
        type_bytes(flow->values[value - 1u].type) != type_bytes(input.type) ||
        (flow->values[value - 1u].count != input.count &&
         !(flow->values[value - 1u].count == 1u && input.count != 1u))) {
      reject(*flow, Reason::GraphShapeMismatch);
      return {};
    }
  }
  std::array<const ExprState *, MaxOutputs> validated_states{};
  std::size_t validated_state_count = 0u;
  for (const ExprRef &expression : expressions) {
    if (expression.state == nullptr || !expression.state->status ||
        expression.node == 0u ||
        expression.node > expression.state->nodes.size() ||
        expression.state->nodes.empty() ||
        (type_bytes(expression.type) != type_bytes(input.type) &&
         !(expressions.size() == 1u &&
           is_width_mask(expression, input.type)))) {
      const Status status = expression.state == nullptr
                                ? Status::fail(Reason::ExpressionCapacity)
                                : expression.state->status;
      reject(*flow, status ? Reason::GraphTypeMismatch : status.reason());
      return {};
    }
    const ExprState *const state = expression.state.get();
    if (std::find(validated_states.begin(),
                  validated_states.begin() + validated_state_count,
                  state) != validated_states.begin() + validated_state_count) {
      continue;
    }
    validated_states[validated_state_count++] = state;
    for (const ExprNode &node : expression.state->nodes) {
      if (node.operation != ExprOp::Input) {
        continue;
      }
      if (node.left >= inputs.size()) {
        reject(*flow, Reason::GraphTypeMismatch);
        return {};
      }
      const FlowValue &source = flow->values[inputs[node.left] - 1u];
      if (node.type != source.type ||
          (type_fixed(node.type) && node.fixed_format != source.fixed_format)) {
        reject(*flow, Reason::GraphTypeMismatch);
        return {};
      }
    }
  }
  ValueIds outputs;
  for (const ExprRef &expression : expressions) {
    const std::uint32_t output =
        append(*flow, expression.type, input.count, expression.fixed_format);
    if (output == 0u) {
      return {};
    }
    outputs.push_back(output);
  }
  if (control.count != 0u) {
    for (const std::uint32_t output : outputs) {
      flow->values[output - 1u].active = control.count;
    }
  }
  return append_map(*flow, name, inputs, outputs, expressions, control)
             ? outputs
             : ValueIds{};
}

std::size_t flow_step_count(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr ? 0u : flow->steps.size();
}

void flow_tag_iteration(const std::shared_ptr<FlowState> &flow,
                        const std::size_t first_step,
                        const std::uint32_t iteration) noexcept {
  if (flow == nullptr || !flow->status || iteration == 0u ||
      first_step > flow->steps.size()) {
    if (flow != nullptr && flow->status) {
      reject(*flow, Reason::BoundedCountInvalid);
    }
    return;
  }
  for (std::size_t index = first_step; index < flow->steps.size(); ++index) {
    FlowStep &step = flow->steps[index];
    if (auto *const map = std::get_if<MapStep>(&step);
        map != nullptr && map->control.count != 0u) {
      map->control.iteration = iteration;
      return;
    }
    if (auto *const scan = std::get_if<ScanStep>(&step);
        scan != nullptr && scan->count != 0u && scan->input != 0u &&
        scan->input <= flow->values.size()) {
      scan->control = FlowControl{
          .count = scan->count,
          .capacity = flow->values[scan->input - 1u].count,
          .iteration = iteration,
      };
      return;
    }
    auto *const primitive = std::get_if<FlowPrimitive>(&step);
    if (primitive == nullptr ||
        (primitive->operation != Primitive::Sort &&
         primitive->operation != Primitive::Argsort) ||
        !flow->value_ids.valid(primitive->inputs)) {
      continue;
    }
    const std::span<const std::uint32_t> inputs =
        flow->value_ids.view(primitive->inputs);
    if (inputs.size() != 2u || inputs.front() == 0u ||
        inputs.front() > flow->values.size()) {
      continue;
    }
    primitive->control = FlowControl{
        .count = inputs.back(),
        .capacity = flow->values[inputs.front() - 1u].count,
        .iteration = iteration,
    };
    return;
  }
  // The body needs one resident count-aware execution point to own its
  // iteration evidence. Map, Scan, and Sort all carry the same canonical
  // resident control; do not manufacture a copy pass merely for telemetry.
  reject(*flow, Reason::BoundedCountInvalid);
}

void flow_map(const std::shared_ptr<FlowState> &flow,
              const std::string_view name, ExprRef expression) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::array inputs{flow->output};
  const std::uint32_t output =
      flow_map_value(flow, inputs, name, std::move(expression));
  if (output != 0u) {
    flow->output = output;
  }
}

std::uint32_t flow_zero(const std::shared_ptr<FlowState> &flow,
                        const std::size_t count) {
  if (flow == nullptr || !flow->status || count == 0u) {
    return 0u;
  }
  const auto expression = make_expr();
  const ExprRef zero = constant(expression, Type::U32, 0u);
  if (zero.node == 0u) {
    reject(*flow, Reason::ExpressionCapacity);
    return 0u;
  }
  const std::uint32_t output = append(*flow, Type::U32, count);
  if (output == 0u) {
    return 0u;
  }
  const std::span<const std::uint32_t> inputs;
  const std::array outputs{output};
  const std::array expressions{zero};
  return append_map(*flow, "zero-index", inputs, outputs, expressions) ? output
                                                                       : 0u;
}

std::uint32_t flow_index(const std::shared_ptr<FlowState> &flow,
                         const Type type, const std::size_t count) {
  if (flow == nullptr || !flow->status || count == 0u) {
    return 0u;
  }
  const auto expression = make_expr();
  const ExprRef ordinal = index(expression, type);
  if (ordinal.node == 0u) {
    reject(*flow, Reason::ExpressionCapacity);
    return 0u;
  }
  const std::uint32_t output = append(*flow, type, count);
  if (output == 0u) {
    return 0u;
  }
  const std::span<const std::uint32_t> inputs;
  const std::array outputs{output};
  const std::array expressions{ordinal};
  return append_map(*flow, "index", inputs, outputs, expressions) ? output : 0u;
}

} // namespace rund::compute::detail
