#include "src/accel/graph/token/reset.hpp"
#include "src/accel/kernel/reset/overlap.hpp"
#include "src/accel/kernel/reset/projection.hpp"
#include "src/accel/kernel/reset/proof.hpp"
#include "src/accel/kernel/run/bindings.hpp"
#include "src/accel/vulkan/kernel/reset.hpp"

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
using rund::node::accel::detail::reset::WordAddressable;

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
  const Result dense32 = Prove(dense, 20u);
  const Params params = Bind(dense32.range, 2u);
  if (!Accepted(dense32) || !dense32.range.dense() ||
      Payload(dense32.range) != 16u || params.count != 4u ||
      dense32.range.end() != 20u ||
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
  const Result sparse = Prove(strided64, 48u);
  const Params sparse_params = Bind(sparse.range, 1u);
  if (!Accepted(sparse) || sparse.range.dense() ||
      Payload(sparse.range) != 24u || sparse.range.end() != 48u ||
      sparse_params.offset_words != 2u ||
      sparse_params.stride_words != 4u || sparse_params.element_words != 2u ||
      Commands(513u, 256u) != 3u || Commands(1u, 0u) != 0u) {
    return false;
  }

  const Replacement replacement{.count = 6u, .element = 8u};
  const Spec projected = Project(source, &replacement);
  const Result replaced = Prove(projected, 48u);
  return projected.offset == 0u && projected.count == 6u &&
         projected.stride == 8u && projected.element == 8u &&
         projected.dense() && Accepted(replaced) &&
         replaced.range.offset() == 0u && replaced.range.end() == 48u;
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
  const Spec shader_strided{
      .offset = shader_limit.offset,
      .count = 1u,
      .stride = 8u,
      .element = 4u,
  };
  const std::uint64_t shader_bytes = shader_limit.offset + 4u;
  const Result shader_range = Prove(shader_limit, shader_bytes);
  const Result shader_strided_range = Prove(shader_strided, shader_bytes);
  const Spec oversized_shader{
      .count = static_cast<std::uint64_t>(UINT32_MAX) + 1u,
      .stride = 4u,
      .element = 4u,
  };
  const std::uint64_t oversized_bytes = oversized_shader.count * 4u;
  const Result oversized_range = Prove(oversized_shader, oversized_bytes);
  return Rejected(Prove(Spec{.offset = 2u,
                             .count = 1u,
                             .stride = 4u,
                             .element = 4u},
                        8u)) &&
         Rejected(Prove(
             Spec{.count = 2u, .stride = 6u, .element = 4u}, 16u)) &&
         Rejected(Prove(
             Spec{.count = 2u, .stride = 4u, .element = 8u}, 16u)) &&
         Rejected(Prove(offset_overflow, maximum)) &&
         Rejected(Prove(stride_overflow, maximum)) &&
         Rejected(Prove(Spec{.offset = 8u,
                             .count = 3u,
                             .stride = 8u,
                             .element = 8u},
                        31u)) &&
         Accepted(shader_range) && shader_range.range.dense() &&
         Accepted(shader_strided_range) &&
         !shader_strided_range.range.dense() &&
         Accepted(oversized_range) &&
         !WordAddressable(shader_range.range, 0u, word_limit) &&
         WordAddressable(shader_range.range, shader_limit.offset, word_limit) &&
         WordAddressable(shader_strided_range.range, shader_limit.offset,
                         word_limit) &&
         !WordAddressable(oversized_range.range, 0u, word_limit) &&
         !WordAddressable(shader_range.range, shader_limit.offset + 4u,
                          maximum) &&
         !WordAddressable(shader_range.range, shader_limit.offset - 1u,
                          maximum) &&
         WordAddressable(shader_range.range, 0u, maximum);
}

[[nodiscard]] bool VulkanExecutionFormIsExact() {
  using rund::node::accel::detail::KernelPreparationMode;
  using rund::node::accel::detail::PlanVulkanResetExecution;

  constexpr std::uint64_t Count = 513u;
  constexpr std::uint64_t Window = 256u;
  const Result dense = Prove(
      Spec{.count = Count, .stride = 4u, .element = 4u}, Count * 4u);
  const Result strided = Prove(
      Spec{.count = Count, .stride = 8u, .element = 4u},
      (Count - 1u) * 8u + 4u);
  if (!Accepted(dense) || !Accepted(strided)) {
    return false;
  }
  const auto standalone_dense = PlanVulkanResetExecution(
      dense.range, KernelPreparationMode::Standalone, Window);
  const auto captured_dense = PlanVulkanResetExecution(
      dense.range, KernelPreparationMode::PipelinePrivate, Window);
  const auto standalone_strided = PlanVulkanResetExecution(
      strided.range, KernelPreparationMode::Standalone, Window);
  const auto dense_without_dispatch = PlanVulkanResetExecution(
      dense.range, KernelPreparationMode::Standalone, 0u);
  const auto captured_without_dispatch = PlanVulkanResetExecution(
      dense.range, KernelPreparationMode::PipelinePrivate, 0u);
  return standalone_dense.ok && !standalone_dense.shader &&
         standalone_dense.commands == 1u && captured_dense.ok &&
         captured_dense.shader && captured_dense.commands == 3u &&
         standalone_strided.ok && standalone_strided.shader &&
         standalone_strided.commands == 3u && dense_without_dispatch.ok &&
         !dense_without_dispatch.shader &&
         dense_without_dispatch.commands == 1u &&
         !captured_without_dispatch.ok;
}

[[nodiscard]] bool LifetimeOverlapIsExact() {
  using rund::node::accel::detail::BoundReset;
  using rund::node::accel::detail::reset::Compatible;

  const auto owner = std::make_shared<std::uint32_t>(0u);
  const auto reset = [&](const std::uint64_t offset, const std::uint32_t first,
                         const std::uint32_t last, const bool external = false,
                         const std::uint64_t count = 4u,
                         const std::uint64_t stride = 4u) {
    const rund::kernel::ResidentBufferRef ref{
        .id = 7u,
        .bytes = 64u,
        .offset_bytes = offset,
        .element_bytes = 4u,
        .stride_bytes = stride,
        .count = count,
        .usage = rund::kernel::kResidentUsageWrite,
    };
    const Result proved = Prove(Project(ref, nullptr), ref.bytes);
    return BoundReset::Seal(
               ref, owner, proved.range, 0u,
               rund::node::accel::detail::ExecStep{first},
               rund::node::accel::detail::ExecStep{last}, external)
        .value();
  };

  const rund::kernel::ResidentBufferRef sealed_source{
      .id = 7u,
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const Result sealed_range =
      Prove(Project(sealed_source, nullptr), sealed_source.bytes);
  auto mismatched_source = sealed_source;
  mismatched_source.offset_bytes = 4u;
  if (BoundReset::Seal(mismatched_source, owner, sealed_range.range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value()) {
    return false;
  }
  auto read_source = sealed_source;
  read_source.usage = rund::kernel::kResidentUsageRead;
  auto unknown_source = sealed_source;
  unknown_source.usage = 0u;
  if (BoundReset::Seal(read_source, owner, sealed_range.range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value() ||
      BoundReset::Seal(unknown_source, owner, sealed_range.range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value()) {
    return false;
  }

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
  const std::array strided_tail_overlap{
      reset(0u, 0u, 2u, false, 3u, 8u),
      reset(16u, 1u, 3u, false, 1u, 4u),
  };

  return Compatible(disjoint_time) && !Compatible(touching_time) &&
         Compatible(adjacent_memory) && !Compatible(external_alias) &&
         !Compatible(strided_tail_overlap);
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
              .usage = rund::kernel::kResidentUsageWrite,
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
  const auto build_with = [&](const std::span<const ResetPlan> resets,
                              const RunBinds &active_bindings) {
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
    return BuildResetBinds(execution, run, active_bindings);
  };
  const auto build = [&](const std::span<const ResetPlan> resets) {
    return build_with(resets, bindings);
  };

  const std::array valid{
      ResetPlan{.binding = 0u, .step = ExecStep{1u}, .last = ExecStep{2u}},
  };
  const auto bound = build(valid);
  const rund::kernel::ResidentBufferRef sealed_ref =
      bound.ok && !bound.binds.empty() ? bound.binds[0u].ref()
                                       : rund::kernel::ResidentBufferRef{};
  if (!bound.ok || bound.binds.size() != 1u ||
      bound.binds[0u].step.index != 1u ||
      bound.binds[0u].last.index != 2u ||
      !bound.binds[0u].range().valid() ||
      bound.binds[0u].range().offset() != 0u ||
      bound.binds[0u].range().count() != 16u ||
      bound.binds[0u].range().end() != 64u ||
      Payload(bound.binds[0u].range()) != 64u ||
      sealed_ref.offset_bytes != bound.binds[0u].range().offset() ||
      sealed_ref.element_bytes != bound.binds[0u].range().element() ||
      sealed_ref.stride_bytes != bound.binds[0u].range().stride() ||
      sealed_ref.count != bound.binds[0u].range().count()) {
    return false;
  }
  const auto rejected_ref = [&](const rund::kernel::ResidentBufferRef ref) {
    RunBinds candidate{};
    return candidate.push(ref, owner) && candidate.valid() &&
           !build_with(valid, candidate).ok;
  };
  auto misaligned_offset = buffer.resident;
  misaligned_offset.offset_bytes = 1u;
  misaligned_offset.count = 4u;
  auto misaligned_stride = buffer.resident;
  misaligned_stride.stride_bytes = 6u;
  misaligned_stride.count = 4u;
  auto unsupported_width = buffer.resident;
  unsupported_width.element_bytes = 2u;
  unsupported_width.count = 4u;
  auto final_byte_overflow = buffer.resident;
  final_byte_overflow.offset_bytes = 52u;
  final_byte_overflow.count = 4u;
  auto read_resident = buffer.resident;
  read_resident.usage = rund::kernel::kResidentUsageRead;
  auto unknown_resident = buffer.resident;
  unknown_resident.usage = 0u;
  if (!rejected_ref(misaligned_offset) || !rejected_ref(misaligned_stride) ||
      !rejected_ref(unsupported_width) ||
      !rejected_ref(final_byte_overflow) || !rejected_ref(read_resident) ||
      !rejected_ref(unknown_resident)) {
    return false;
  }
  auto trailing_owner_bytes = buffer.resident;
  trailing_owner_bytes.bytes = 5u;
  trailing_owner_bytes.count = 1u;
  RunBinds trailing_bindings{};
  if (!trailing_bindings.push(trailing_owner_bytes, owner) ||
      !trailing_bindings.valid()) {
    return false;
  }
  const auto trailing = build_with(valid, trailing_bindings);
  if (!trailing.ok || trailing.binds.size() != 1u ||
      trailing.binds[0u].range().end() != 4u) {
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
         VulkanExecutionFormIsExact() && ProjectionIsUnique() &&
         LifetimeOverlapIsExact() &&
         ResetPlansAreSealed() && ProjectionMatrixIsExact() &&
         ProjectionRejectsAmbiguity();
}

} // namespace node_accel_contract
