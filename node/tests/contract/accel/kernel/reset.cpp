#include "src/accel/graph/token/reset.hpp"
#include "src/accel/kernel/reset/overlap.hpp"
#include "src/accel/kernel/reset/projection.hpp"
#include "src/accel/kernel/reset/proof.hpp"
#include "src/accel/kernel/run/bindings.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace node_accel_contract {
namespace {

using rund::node::accel::detail::reset::Bind;
using rund::node::accel::detail::reset::Commands;
using rund::node::accel::detail::reset::Find;
using rund::node::accel::detail::reset::Params;
using rund::node::accel::detail::reset::Payload;
using rund::node::accel::detail::reset::Project;
using rund::node::accel::detail::reset::Prove;
using rund::node::accel::detail::reset::Replacement;
using rund::node::accel::detail::reset::Result;
using rund::node::accel::detail::reset::Spec;

[[nodiscard]] bool Accepted(const Result result) noexcept {
  return result.check.ok && std::string_view{result.check.reason} == "ok";
}

[[nodiscard]] bool Rejected(const Result result) noexcept {
  return !result.check.ok &&
         std::string_view{result.check.reason} == "accel_kernel_reset_invalid";
}

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

[[nodiscard]] bool ProjectionIsUnique() {
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

[[nodiscard]] bool WidthAndLayoutProofsMatch() {
  const rund::kernel::ResidentBufferRef source{
      .bytes = 48u,
      .offset_bytes = 4u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
  };
  const Spec dense = Project(source, nullptr);
  const Result dense32 = Prove(dense, 20u, UINT64_MAX);
  const Params params = Bind(dense32.range, 2u);
  if (!Accepted(dense32) || !dense32.range.dense() ||
      Payload(dense32.range) != 16u || params.count != 4u ||
      params.base != 2u || params.offset_words != 1u ||
      params.stride_words != 1u || params.element_words != 1u ||
      sizeof(params) != 40u) {
    return false;
  }

  const Spec strided64{
      .offset = 8u,
      .count = 3u,
      .stride = 16u,
      .element = 8u,
  };
  const Result sparse = Prove(strided64, 48u, UINT64_MAX);
  const Params sparse_params = Bind(sparse.range, 1u);
  if (!Accepted(sparse) || sparse.range.dense() ||
      Payload(sparse.range) != 24u || sparse_params.offset_words != 2u ||
      sparse_params.stride_words != 4u || sparse_params.element_words != 2u ||
      Commands(513u, 256u) != 3u || Commands(1u, 0u) != 0u) {
    return false;
  }

  const Replacement replacement{.count = 6u, .element = 8u};
  const Spec projected = Project(source, &replacement);
  return projected.offset == 0u && projected.count == 6u &&
         projected.stride == 8u && projected.element == 8u && projected.dense();
}

[[nodiscard]] bool InvalidRangesHaveOneReason() {
  constexpr std::uint64_t maximum = UINT64_MAX;
  constexpr std::uint64_t word_limit = UINT32_MAX;
  const Spec offset_overflow{
      .offset = maximum - 3u,
      .count = 1u,
      .stride = 8u,
      .element = 8u,
  };
  const Spec stride_overflow{
      .count = 3u,
      .stride = maximum - 3u,
      .element = 4u,
  };
  const Spec shader_limit{
      .offset = static_cast<std::uint64_t>(UINT32_MAX) * 4u,
      .count = 1u,
      .stride = 4u,
      .element = 4u,
  };
  const std::uint64_t shader_bytes = shader_limit.offset + 4u;
  return Rejected(
             Prove(Spec{.offset = 2u, .count = 1u, .stride = 4u, .element = 4u},
                   8u, maximum)) &&
         Rejected(Prove(Spec{.count = 2u, .stride = 6u, .element = 4u}, 16u,
                        maximum)) &&
         Rejected(Prove(Spec{.count = 2u, .stride = 4u, .element = 8u}, 16u,
                        maximum)) &&
         Rejected(Prove(offset_overflow, maximum, maximum)) &&
         Rejected(Prove(stride_overflow, maximum, maximum)) &&
         Rejected(
             Prove(Spec{.offset = 8u, .count = 3u, .stride = 8u, .element = 8u},
                   31u, maximum)) &&
         Rejected(Prove(shader_limit, shader_bytes, word_limit)) &&
         Accepted(Prove(shader_limit, shader_bytes, maximum));
}

[[nodiscard]] bool LifetimeOverlapIsExact() {
  using rund::node::accel::detail::BoundReset;
  using rund::node::accel::detail::reset::Compatible;

  const auto owner = std::make_shared<std::uint32_t>(0u);
  const auto reset = [&](const std::uint64_t offset, const std::uint32_t first,
                         const std::uint32_t last,
                         const bool external = false) {
    return BoundReset{
        .ref =
            {
                .id = 7u,
                .bytes = 64u,
                .offset_bytes = offset,
                .element_bytes = 4u,
                .stride_bytes = 4u,
                .count = 4u,
            },
        .handle = owner,
        .step = rund::node::accel::detail::ExecStep{first},
        .last = rund::node::accel::detail::ExecStep{last},
        .external = external,
    };
  };

  const std::array disjoint_time{
      reset(0u, 0u, 1u),
      reset(8u, 2u, 3u),
  };
  const std::array touching_time{
      reset(0u, 0u, 1u),
      reset(8u, 1u, 2u),
  };
  const std::array adjacent_memory{
      reset(0u, 0u, 2u),
      reset(16u, 1u, 3u),
  };
  const std::array external_alias{
      reset(0u, 0u, 1u, true),
      reset(8u, 2u, 3u),
  };

  return Compatible(disjoint_time) && !Compatible(touching_time) &&
         Compatible(adjacent_memory) && !Compatible(external_alias);
}

[[nodiscard]] bool ResetPlansAreSealed() {
  using rund::node::accel::detail::BuildResetBinds;
  using rund::node::accel::detail::ExecStep;
  using rund::node::accel::detail::KernelExecution;
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::ResetPlan;
  using rund::node::accel::detail::RunBinds;

  const auto owner = std::make_shared<std::uint32_t>(0u);
  const rund::AccelBuffer buffer{
      .resident =
          {
              .id = 11u,
              .bytes = 64u,
              .element_bytes = 4u,
              .stride_bytes = 4u,
              .count = 16u,
          },
      .byte_extent = 64u,
      .usage = rund::BufferUsage::WriteOnly,
  };
  const std::array<rund::AccelRunBinding, 1u> run_bindings{
      rund::AccelRunBinding{
          .buffer = &buffer,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  RunBinds bindings{};
  if (!bindings.push(buffer.resident, owner) || !bindings.valid()) {
    return false;
  }
  const std::array roles{rund::kernel::BufferRole::Write};
  const std::array visibilities{rund::GraphBufferVisibility::Internal};
  const std::array<std::uint64_t, 1u> aliases{0u};
  std::array<KernelExecutionStep, 3u> steps{};
  const auto build = [&](const std::span<const ResetPlan> resets) {
    const KernelExecution execution{
        .graph_roles = roles,
        .graph_visibilities = visibilities,
        .graph_alias_representatives = aliases,
        .resets = resets,
        .steps = steps,
    };
    const rund::AccelRun run{
        .bindings = run_bindings.data(),
        .binding_count = run_bindings.size(),
    };
    return BuildResetBinds(execution, run, bindings);
  };

  const std::array valid{
      ResetPlan{.binding = 0u, .step = ExecStep{1u}, .last = ExecStep{2u}},
  };
  const auto bound = build(valid);
  if (!bound.ok || bound.binds.size() != 1u ||
      bound.binds[0u].step.index != 1u || bound.binds[0u].last.index != 2u) {
    return false;
  }
  const std::array reversed{
      ResetPlan{.binding = 0u, .step = ExecStep{2u}, .last = ExecStep{1u}},
  };
  const std::array missing{
      ResetPlan{.binding = 0u},
  };
  const std::array duplicate{
      ResetPlan{.binding = 0u, .step = ExecStep{1u}, .last = ExecStep{2u}},
      ResetPlan{.binding = 0u, .step = ExecStep{1u}, .last = ExecStep{2u}},
  };
  return !build(reversed).ok && !build(missing).ok && !build(duplicate).ok;
}

[[nodiscard]] bool ProjectionMatrixIsExact() {
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

[[nodiscard]] bool ProjectionRejectsAmbiguity() {
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

} // namespace

bool ResetModelContract() {
  return WidthAndLayoutProofsMatch() && InvalidRangesHaveOneReason() &&
         ProjectionIsUnique() && LifetimeOverlapIsExact() &&
         ResetPlansAreSealed() && ProjectionMatrixIsExact() &&
         ProjectionRejectsAmbiguity();
}

} // namespace node_accel_contract
