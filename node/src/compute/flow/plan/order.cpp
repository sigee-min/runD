#include "state.hpp"

#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Status
canonical_step_order(const FlowState &flow, const std::vector<bool> &keep,
                     const std::vector<MapLivePlan> &map_plans,
                     const std::size_t live_steps,
                     std::vector<std::size_t> &order) {
  constexpr std::size_t NoProducer = std::numeric_limits<std::size_t>::max();
  try {
    std::vector<std::size_t> producer(flow.values.size() + 1u, NoProducer);
    std::vector<bool> external(flow.values.size() + 1u, false);
    std::vector<std::uint8_t> state(flow.steps.size(), 0u);
    order.clear();
    order.reserve(live_steps);

    for (const std::uint32_t input : flow.inputs) {
      if (input == 0u || input > flow.values.size() || external[input]) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      external[input] = true;
    }
    const auto register_output = [&](const std::uint32_t output,
                                     const std::size_t step) {
      if (output == 0u || output > flow.values.size() || external[output] ||
          producer[output] != NoProducer) {
        return false;
      }
      producer[output] = step;
      return true;
    };
    for (std::size_t step = 0u; step < flow.steps.size(); ++step) {
      if (!keep[step]) {
        continue;
      }
      const FlowStep &entry = flow.steps[step];
      if (const auto *map = std::get_if<MapStep>(&entry)) {
        const std::span<const std::uint32_t> outputs =
            flow.value_ids.view(map->outputs);
        for (const std::uint32_t output : outputs) {
          if (!register_output(output, step)) {
            return Status::fail(Reason::GraphBindingInvalid);
          }
        }
      } else if (const auto *scan = std::get_if<ScanStep>(&entry)) {
        if (!register_output(scan->output, step)) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
      } else {
        const auto &primitive = std::get<FlowPrimitive>(entry);
        const std::span<const std::uint32_t> outputs =
            flow.value_ids.view(primitive.outputs);
        for (const std::uint32_t output : outputs) {
          if (!register_output(output, step)) {
            return Status::fail(Reason::GraphBindingInvalid);
          }
        }
      }
    }

    const auto visit_step = [&](auto &&self, const std::size_t step) -> bool {
      if (step >= flow.steps.size() || !keep[step]) {
        return false;
      }
      if (state[step] == 2u) {
        return true;
      }
      if (state[step] == 1u) {
        return false;
      }
      state[step] = 1u;
      const auto visit_value = [&](const std::uint32_t value) {
        if (value == 0u || value > flow.values.size()) {
          return false;
        }
        std::uint32_t guard = flow.values[value - 1u].guard;
        while (guard != 0u) {
          if (!external[guard] &&
              (producer[guard] == NoProducer || !self(self, producer[guard]))) {
            return false;
          }
          guard = flow.values[guard - 1u].guard;
        }
        if (external[value]) {
          return true;
        }
        return producer[value] != NoProducer && self(self, producer[value]);
      };

      const FlowStep &entry = flow.steps[step];
      if (const auto *map = std::get_if<MapStep>(&entry)) {
        const std::span<const std::uint32_t> map_inputs =
            flow.value_ids.view(map->inputs);
        const MapLivePlan &map_plan = map_plans[step];
        for (std::size_t input = 0u; input < map_inputs.size(); ++input) {
          if ((map_plan.used_inputs & live_bit(input)) != 0u &&
              !visit_value(map_inputs[input])) {
            return false;
          }
        }
        if ((map->control.count != 0u && !visit_value(map->control.count)) ||
            (map->control.predicate != 0u &&
             !visit_value(map->control.predicate))) {
          return false;
        }
      } else if (const auto *scan = std::get_if<ScanStep>(&entry)) {
        if (!visit_value(scan->input) ||
            (scan->count != 0u && !visit_value(scan->count))) {
          return false;
        }
      } else {
        const auto &primitive = std::get<FlowPrimitive>(entry);
        const std::span<const std::uint32_t> inputs =
            flow.value_ids.view(primitive.inputs);
        for (const std::uint32_t input : inputs) {
          if (!visit_value(input)) {
            return false;
          }
        }
      }
      state[step] = 2u;
      order.push_back(step);
      return true;
    };
    const auto visit_output = [&](const std::uint32_t value) {
      if (value == 0u || value > flow.values.size()) {
        return false;
      }
      std::uint32_t guard = flow.values[value - 1u].guard;
      while (guard != 0u) {
        if (!external[guard] && (producer[guard] == NoProducer ||
                                 !visit_step(visit_step, producer[guard]))) {
          return false;
        }
        guard = flow.values[guard - 1u].guard;
      }
      return external[value] || (producer[value] != NoProducer &&
                                 visit_step(visit_step, producer[value]));
    };

    const std::span<const std::uint32_t> selected =
        flow.outputs.empty() ? std::span<const std::uint32_t>{&flow.output, 1u}
                             : std::span<const std::uint32_t>{flow.outputs};
    const std::span<const std::uint32_t> logical =
        flow.logical_outputs.empty()
            ? selected
            : std::span<const std::uint32_t>{flow.logical_outputs};
    for (const std::uint32_t output : logical) {
      if (!visit_output(output)) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
    }
    for (const std::uint32_t output : selected) {
      if (!visit_output(output)) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
    }
    if (order.size() != live_steps) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail
