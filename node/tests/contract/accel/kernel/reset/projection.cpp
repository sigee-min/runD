#include "local.hpp"

#include "src/accel/graph/token/reset.hpp"
#include "src/accel/kernel/reset/projection.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace node_accel_contract::reset_contract {
namespace {

using rund::node::accel::detail::reset::Find;

struct Transfer final {
  bool input{};
};

struct View final {
  std::array<Transfer, 2u> transfers{};
  std::array<std::uint32_t, 1u> transfer_by_binding{};
};

struct Entry final {
  View *view{};
};

struct Resources final {
  std::array<Entry, 2u> entries{};
  std::size_t count{};

  [[nodiscard]] std::size_t size() const noexcept { return count; }

  [[nodiscard]] Entry *entry(const std::size_t index) noexcept {
    return index < count ? &entries[index] : nullptr;
  }
};

} // namespace

bool CheckUniqueProjection() {
  View first{};
  View second{};
  first.transfer_by_binding[0u] = 1u;
  Resources resources{.entries = {{{.view = &first}, {.view = &second}}},
                      .count = 2u};
  const Transfer *replacement = nullptr;
  if (!Find(resources, 0u, replacement) ||
      replacement != &first.transfers[0u]) {
    return false;
  }

  second.transfer_by_binding[0u] = 1u;
  if (Find(resources, 0u, replacement) || replacement != nullptr) {
    return false;
  }
  second.transfer_by_binding[0u] = 0u;
  first.transfers[0u].input = true;
  if (Find(resources, 0u, replacement) || replacement != nullptr) {
    return false;
  }
  first.transfers[0u].input = false;
  first.transfer_by_binding[0u] = 3u;
  if (Find(resources, 0u, replacement) || replacement != nullptr) {
    return false;
  }
  first.transfer_by_binding[0u] = 0u;
  return Find(resources, 0u, replacement) && replacement == nullptr;
}

bool CheckProjectionMatrix() {
  using rund::node::accel::detail::ExecStep;
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::PlanResets;
  using rund::node::accel::detail::ResetPlan;
  using rund::node::accel::detail::SourceRange;
  using rund::node::accel::detail::SourceStep;

  constexpr std::uint32_t Limit = 7u;
  for (std::uint32_t count = 1u; count <= Limit; ++count) {
    const std::uint32_t partition_count = 1u << (count - 1u);
    for (std::uint32_t partition = 0u; partition < partition_count;
         ++partition) {
      std::vector<KernelExecutionStep> steps;
      std::array<ExecStep, Limit> owner{};
      std::uint32_t begin = 0u;
      for (std::uint32_t source = 0u; source < count; ++source) {
        const bool closes =
            source + 1u == count || (partition & (1u << source)) != 0u;
        if (!closes) {
          continue;
        }
        KernelExecutionStep step{};
        step.source = SourceRange{
            .begin = SourceStep{begin},
            .end = SourceStep{source + 1u},
        };
        for (std::uint32_t binding = begin; binding <= source; ++binding) {
          if (!step.graph_binding_indices.push_back(binding)) {
            return false;
          }
          owner[binding] = ExecStep{static_cast<std::uint32_t>(steps.size())};
        }
        step.graph_binding_indices_ok = step.graph_binding_indices.valid();
        steps.push_back(std::move(step));
        begin = source + 1u;
      }
      if (!rund::node::accel::detail::ValidSourcePartition(steps, count)) {
        return false;
      }

      for (std::uint32_t reset = 0u; reset < count; ++reset) {
        std::array<rund::kernel::BufferRole, Limit> roles{};
        std::array<rund::GraphBufferVisibility, Limit> visibilities{};
        std::array<std::uint64_t, Limit> aliases{};
        std::array<SourceStep, Limit> sources{};
        for (std::uint32_t binding = 0u; binding < count; ++binding) {
          roles[binding] = binding == reset ? rund::kernel::BufferRole::Write
                                            : rund::kernel::BufferRole::Read;
          visibilities[binding] = rund::GraphBufferVisibility::External;
          aliases[binding] = binding < reset ? binding : reset;
          sources[binding] = SourceStep{binding};
        }
        const std::array<std::uint64_t, 1u> reset_bindings{reset};
        std::vector<ResetPlan> plans;
        if (!PlanResets(steps, std::span{roles}.first(count),
                        std::span{visibilities}.first(count),
                        std::span{aliases}.first(count),
                        std::span{sources}.first(count), reset_bindings,
                        plans) ||
            plans.size() != 1u || plans.front().binding != reset ||
            plans.front().step.index != owner[reset].index ||
            plans.front().last.index != owner[count - 1u].index) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckProjectionAmbiguity() {
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::PlanResets;
  using rund::node::accel::detail::ResetPlan;
  using rund::node::accel::detail::SourceRange;
  using rund::node::accel::detail::SourceStep;

  std::array<KernelExecutionStep, 2u> steps{};
  for (std::uint32_t index = 0u; index < steps.size(); ++index) {
    steps[index].source = SourceRange{
        .begin = SourceStep{index},
        .end = SourceStep{index + 1u},
    };
    if (!steps[index].graph_binding_indices.push_back(index)) {
      return false;
    }
    steps[index].graph_binding_indices_ok =
        steps[index].graph_binding_indices.valid();
  }
  const std::array base_roles{rund::kernel::BufferRole::Write,
                              rund::kernel::BufferRole::Read};
  const std::array base_visibilities{
      rund::GraphBufferVisibility::External,
      rund::GraphBufferVisibility::External,
  };
  const std::array<std::uint64_t, 2u> base_aliases{0u, 0u};
  const std::array base_sources{SourceStep{0u}, SourceStep{1u}};
  const std::array<std::uint64_t, 1u> reset{0u};
  std::vector<ResetPlan> plans;
  if (!PlanResets(steps, base_roles, base_visibilities, base_aliases,
                  base_sources, reset, plans) ||
      plans.size() != 1u || plans.front().step.index != 0u ||
      plans.front().last.index != 1u) {
    return false;
  }

  auto read_first = base_roles;
  read_first[0u] = rund::kernel::BufferRole::Read;
  auto bad_sources = base_sources;
  bad_sources[1u] = SourceStep{0u};
  const std::array<std::uint64_t, 2u> bad_aliases{1u, 0u};
  const std::array<std::uint64_t, 2u> duplicate_reset{0u, 0u};
  if (PlanResets(steps, read_first, base_visibilities, base_aliases,
                 base_sources, reset, plans) ||
      PlanResets(steps, base_roles, base_visibilities, base_aliases,
                 bad_sources, reset, plans) ||
      PlanResets(steps, base_roles, base_visibilities, bad_aliases,
                 base_sources, reset, plans) ||
      PlanResets(steps, base_roles, base_visibilities, base_aliases,
                 base_sources, duplicate_reset, plans)) {
    return false;
  }

  std::array<KernelExecutionStep, 1u> survivor{};
  survivor[0u].source =
      SourceRange{.begin = SourceStep{0u}, .end = SourceStep{1u}};
  if (!survivor[0u].graph_binding_indices.push_back(1u)) {
    return false;
  }
  survivor[0u].graph_binding_indices_ok =
      survivor[0u].graph_binding_indices.valid();
  const std::array fused_roles{rund::kernel::BufferRole::Write,
                               rund::kernel::BufferRole::Read};
  std::array fused_visibilities{
      rund::GraphBufferVisibility::Internal,
      rund::GraphBufferVisibility::External,
  };
  const std::array fused_sources{SourceStep{0u}, SourceStep{0u}};
  const std::array<std::uint64_t, 2u> independent{0u, 1u};
  if (!PlanResets(survivor, fused_roles, fused_visibilities, independent,
                  fused_sources, reset, plans) ||
      !plans.empty()) {
    return false;
  }
  fused_visibilities[0u] = rund::GraphBufferVisibility::External;
  if (PlanResets(survivor, fused_roles, fused_visibilities, independent,
                 fused_sources, reset, plans)) {
    return false;
  }
  fused_visibilities[0u] = rund::GraphBufferVisibility::Internal;
  const std::array<std::uint64_t, 2u> shared{0u, 0u};
  return !PlanResets(survivor, fused_roles, fused_visibilities, shared,
                     fused_sources, reset, plans);
}

} // namespace node_accel_contract::reset_contract
