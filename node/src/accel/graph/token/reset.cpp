#include "reset.hpp"

#include "../../kernel/backend/exception.hpp"

namespace rund::node::accel::detail {
namespace {

struct AliasSteps final {
  SourceStep first{};
  SourceStep last{};
  ExecStep last_step{};
  bool first_read{};
  bool claimed{};
};

static_assert(sizeof(AliasSteps) == 16u);

} // namespace

bool PlanResets(const std::span<const KernelExecutionStep> steps,
                const std::span<const rund::kernel::BufferRole> roles,
                const std::span<const rund::GraphBufferVisibility> visibilities,
                const std::span<const std::uint64_t> aliases,
                const std::span<const SourceStep> sources,
                const std::span<const std::uint64_t> reset_bindings,
                std::vector<ResetPlan> &plans) noexcept try {
  if (steps.size() > ExecStep::none || roles.size() != aliases.size() ||
      visibilities.size() != aliases.size() ||
      sources.size() != aliases.size()) {
    return false;
  }
  std::vector<ExecStep> owners(aliases.size());
  std::vector<AliasSteps> projected(aliases.size());
  for (std::size_t step = 0u; step < steps.size(); ++step) {
    const KernelExecutionStep &source = steps[step];
    if (!source.graph_binding_indices_ok) {
      return false;
    }
    for (std::size_t local = 0u; local < source.graph_binding_indices.size();
         ++local) {
      const std::uint64_t binding = source.graph_binding_indices[local];
      if (binding >= aliases.size() || owners[binding].valid() ||
          !sources[binding].valid() ||
          !source.source.contains(sources[binding])) {
        return false;
      }
      const std::uint64_t representative = aliases[binding];
      if (representative >= aliases.size() ||
          aliases[representative] != representative) {
        return false;
      }
      const ExecStep execution{static_cast<std::uint32_t>(step)};
      owners[binding] = execution;
      AliasSteps &alias = projected[representative];
      if (!alias.first.valid() || sources[binding].index < alias.first.index) {
        alias.first = sources[binding];
        alias.first_read = roles[binding] == rund::kernel::BufferRole::Read;
      } else if (alias.first.index == sources[binding].index &&
                 roles[binding] == rund::kernel::BufferRole::Read) {
        alias.first_read = true;
      }
      if (!alias.last.valid() || alias.last.index < sources[binding].index) {
        alias.last = sources[binding];
        alias.last_step = execution;
      } else if (alias.last.index == sources[binding].index &&
                 alias.last_step.index != execution.index) {
        return false;
      }
    }
  }
  plans.clear();
  plans.reserve(reset_bindings.size());
  for (const std::uint64_t binding : reset_bindings) {
    if (binding >= aliases.size() ||
        roles[binding] != rund::kernel::BufferRole::Write ||
        !sources[binding].valid()) {
      return false;
    }
    const std::uint64_t representative = aliases[binding];
    if (representative >= aliases.size() ||
        aliases[representative] != representative) {
      return false;
    }
    AliasSteps &alias = projected[representative];
    if (alias.claimed) {
      return false;
    }
    alias.claimed = true;
    if (!owners[binding].valid()) {
      if (visibilities[binding] != rund::GraphBufferVisibility::Internal ||
          alias.first.valid()) {
        return false;
      }
      continue;
    }
    if (!alias.first.valid() || alias.first.index != sources[binding].index ||
        alias.first_read) {
      return false;
    }
    const ExecStep last = alias.last_step;
    if (!last.valid() || last.index < owners[binding].index) {
      return false;
    }
    const ResetPlan plan{
        .binding = binding,
        .step = owners[binding],
        .last = last,
    };
    if (!plans.empty() && (plans.back().step.index > plan.step.index ||
                           (plans.back().step.index == plan.step.index &&
                            plans.back().binding >= plan.binding))) {
      return false;
    }
    plans.push_back(plan);
  }
  return true;
} catch (...) {
  backend_exception::RethrowUnlessCapacityException();
  return false;
}

} // namespace rund::node::accel::detail
