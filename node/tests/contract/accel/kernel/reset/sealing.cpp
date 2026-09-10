#include "local.hpp"

#include "src/accel/graph/token/reset.hpp"
#include "src/accel/kernel/reset/model.hpp"
#include "src/accel/kernel/run/bindings.hpp"

#include <array>
#include <memory>
#include <span>
#include <string_view>

namespace node_accel_contract::reset_contract {

bool CheckSealedPlans() {
  using rund::node::accel::detail::BuildResetBinds;
  using rund::node::accel::detail::ExecStep;
  using rund::node::accel::detail::KernelExecution;
  using rund::node::accel::detail::KernelExecutionStep;
  using rund::node::accel::detail::ResetPlan;
  using rund::node::accel::detail::RunBinds;
  using rund::node::accel::detail::reset::Payload;

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
      bound.binds[0u].step.index != 1u || bound.binds[0u].last.index != 2u ||
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
    if (!candidate.push(ref, owner) || !candidate.valid()) {
      return false;
    }
    const auto rejected = build_with(valid, candidate);
    return !rejected.ok &&
           std::string_view{rejected.reason} == "accel_kernel_reset_invalid";
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
      !rejected_ref(unsupported_width) || !rejected_ref(final_byte_overflow) ||
      !rejected_ref(read_resident) || !rejected_ref(unknown_resident)) {
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

} // namespace node_accel_contract::reset_contract
