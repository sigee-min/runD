#include "dispatch.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

// local.hpp owns the compact aliases consumed by the preparation fragments.
// clang-format off
#include "prepare/local.hpp"
#include "prepare/caps.hpp"
#include "prepare/bindings.hpp"
#include "prepare/plan.hpp"
// clang-format on

namespace rund::node::accel::cpu_simd_detail {

PreparedRun
PrepareRun(const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
           const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
           const rund::kernel::BindingSet &bindings) {
  if (!SupportsPortableLaneShape(caps, ir.scalar)) {
    return RejectPrepared("cpu_simd_lane_width_unsupported");
  }

  PreparedRun prepared;
  prepared.domain = ir.domain;
  prepared.strategy = caps.strategy;
  const BindingPlan binding_plan = BuildBindingPlan(parsed);
  if (const char *const reason =
          ValidateBindings(ir, parsed, binding_plan, bindings);
      reason != nullptr) {
    return RejectPrepared(reason);
  }
  if (const char *const reason = BuildPreparedPlan(
          parsed, binding_plan, bindings, ScalarBytes(ir.scalar), prepared);
      reason != nullptr) {
    return RejectPrepared(reason);
  }
  prepared.ok = true;
  prepared.reason = "ok";
  return prepared;
}

namespace {

[[nodiscard]] CpuSimdDispatch
RejectCpuSimdDispatch(const char *const reason) noexcept {
  CpuSimdDispatch dispatch{};
  dispatch.prepared.reason = reason;
  return dispatch;
}

[[nodiscard]] CpuSimdDispatch PrepareAdmittedCpuSimdDispatch(
    const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::BindingSet &bindings) {
  CpuSimdDispatch dispatch{.prepared = PrepareRun(ir, caps, parsed, bindings)};
  if (!dispatch.prepared.ok) {
    return dispatch;
  }
  dispatch.run = SelectCpuSimdRunner(ir.scalar);
  dispatch.scratch_bytes = SelectCpuSimdScratchSizer(ir.scalar);
  if (dispatch.run == nullptr || dispatch.scratch_bytes == nullptr) {
    dispatch.prepared.ok = false;
    dispatch.prepared.reason = "cpu_simd_scalar_unsupported";
  }
  return dispatch;
}

[[nodiscard]] const char *
CpuCapsFailure(const rund::kernel::CpuCaps &caps) noexcept {
  if (!caps.ok) {
    return caps.reason;
  }
  return rund::kernel::CpuCapsValid(caps) ? nullptr : "cpu_caps_invalid";
}

[[nodiscard]] const char *GenericCpuInputFailure(
    const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input) {
  if (const char *const reason = CpuCapsFailure(caps); reason != nullptr) {
    return reason;
  }
  if (!input.ok) {
    return input.reason;
  }
  return input.key.api == rund::kernel::ComputeApi::Cpu &&
                 input.key.scalar == ir.scalar &&
                 input.key.domain == ir.domain &&
                 input.key.fixed_format == ir.fixed_format &&
                 input.key.op_hash_hi == ir.op_hash_hi &&
                 input.key.op_hash_lo == ir.op_hash_lo
             ? nullptr
             : "compute_artifact_mismatch";
}

} // namespace

CpuSimdDispatch
PrepareCpuSimdDispatch(const rund::kernel::ComputeIR &ir,
                       const rund::kernel::CpuCaps &caps,
                       const rund::kernel::BindingSet &bindings) {
  if (const char *const reason = CpuCapsFailure(caps); reason != nullptr) {
    return RejectCpuSimdDispatch(reason);
  }
  auto input = rund::kernel::compute_lowering_detail::AdmitComputeInput(
      ir, rund::kernel::ComputeApi::Cpu);
  if (!input.ok) {
    return RejectCpuSimdDispatch(input.reason);
  }
  return PrepareAdmittedCpuSimdDispatch(ir, caps, input.parsed, bindings);
}

CpuSimdDispatch PrepareCpuSimdDispatch(
    const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const rund::kernel::BindingSet &bindings) {
  if (const char *const reason = GenericCpuInputFailure(ir, caps, input);
      reason != nullptr) {
    return RejectCpuSimdDispatch(reason);
  }
  return PrepareAdmittedCpuSimdDispatch(ir, caps, input.parsed, bindings);
}

CpuSimdDispatch
PrepareCpuSimdDispatch(const rund::kernel::ComputeIR &ir,
                       const rund::kernel::CpuCaps &caps,
                       const rund::kernel::LoweringArtifact &artifact,
                       const rund::kernel::BindingSet &bindings) {
  if (const char *const reason = CpuCapsFailure(caps); reason != nullptr) {
    return RejectCpuSimdDispatch(reason);
  }
  auto admission = rund::kernel::compute_lowering_detail::AdmitArtifact(
      ir, rund::kernel::ComputeApi::Cpu, artifact);
  if (!admission.ok) {
    return RejectCpuSimdDispatch(admission.reason);
  }
  return PrepareAdmittedCpuSimdDispatch(ir, caps, admission.input.parsed,
                                        bindings);
}

} // namespace rund::node::accel::cpu_simd_detail
