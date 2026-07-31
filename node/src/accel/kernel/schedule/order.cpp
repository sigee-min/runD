#include "../schedule.hpp"

#include <new>

namespace rund::node::accel::detail {

ScheduledStepOrder BuildScheduledStepOrder(
    const std::span<const KernelExecutionStep> steps,
    const std::span<const rund::kernel::BufferRole> graph_roles,
    const std::span<const std::uint64_t> graph_alias_representatives,
    const std::span<const std::uint8_t> required_barriers) {
  if (graph_roles.size() != graph_alias_representatives.size() ||
      required_barriers.size() != steps.size()) {
    return ScheduledStepOrder{.ok = false};
  }
  try {
    ScheduledStepOrder result{
        .barriers = std::vector<std::uint8_t>(steps.size(), 0u),
        .count = steps.size(),
    };
    std::vector<std::uint8_t> pending(graph_roles.size(), 0u);
    std::vector<std::size_t> touched;
    touched.reserve(graph_roles.size());
    for (std::size_t step_index = 0u; step_index < steps.size(); ++step_index) {
      const KernelExecutionStep &step = steps[step_index];
      if (!step.graph_binding_indices_ok) {
        return ScheduledStepOrder{.ok = false};
      }
      bool barrier = required_barriers[step_index] != 0u;
      if (barrier) {
        for (const std::size_t representative : touched) {
          pending[representative] = 0u;
        }
        touched.clear();
      }
      for (std::uint64_t local = 0u; local < step.graph_binding_indices.size();
           ++local) {
        const std::uint64_t binding = step.graph_binding_indices[local];
        if (binding >= graph_roles.size()) {
          return ScheduledStepOrder{.ok = false};
        }
        const std::uint64_t representative =
            graph_alias_representatives[binding];
        const rund::kernel::BufferRole role = graph_roles[binding];
        if (representative >= pending.size() ||
            (role != rund::kernel::BufferRole::Read &&
             role != rund::kernel::BufferRole::Write)) {
          return ScheduledStepOrder{.ok = false};
        }
        const std::uint8_t prior =
            pending[static_cast<std::size_t>(representative)];
        barrier = barrier ||
                  (prior != 0u &&
                   (prior == 2u || role == rund::kernel::BufferRole::Write));
      }
      if (barrier) {
        result.barriers[step_index] = 1u;
        for (const std::size_t representative : touched) {
          pending[representative] = 0u;
        }
        touched.clear();
      }
      for (std::uint64_t local = 0u; local < step.graph_binding_indices.size();
           ++local) {
        const std::uint64_t binding = step.graph_binding_indices[local];
        const std::size_t representative =
            static_cast<std::size_t>(graph_alias_representatives[binding]);
        std::uint8_t &mode = pending[representative];
        if (mode == 0u) {
          touched.push_back(representative);
        }
        if (graph_roles[binding] == rund::kernel::BufferRole::Write) {
          mode = 2u;
        } else if (mode == 0u) {
          mode = 1u;
        }
      }
    }
    return result;
  } catch (const std::bad_alloc &) {
    return ScheduledStepOrder{.ok = false};
  }
}

} // namespace rund::node::accel::detail
