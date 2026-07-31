#include "local.hpp"

#include "buffer.hpp"
#include "simd/dispatch.hpp"
#include "simd/run/index.hpp"

#include "../plan/validation.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <cstddef>
#include <vector>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool
ValidateCpuMap(CpuAdapter &adapter, const rund::kernel::ComputePlan &plan,
               const rund::kernel::ExecutionMetadata &metadata,
               const rund::kernel::ComputeDispatchWindow *const windows,
               const rund::kernel::u64 window_count,
               const rund::kernel::BindingSet &bindings) {
  if (!ComputePlanHeaderValid(plan, rund::kernel::ComputeApi::Cpu) ||
      !FrozenCapsAdmitPlan(adapter.generic_caps, plan) ||
      !CpuWindowsMatchPlan(plan, windows, window_count)) {
    adapter.last_error = "accel_cpu_backend_invalid";
    return false;
  }

  const rund::kernel::BindingValidation validation = BindingPlanCheck(
      plan, bindings, metadata, PlanBindingInputMode::StagedOrResident);
  if (!validation.ok) {
    adapter.last_error = validation.reason;
    return false;
  }
  if (bindings.has_resident_output() ||
      bindings.resident_inputs.refs != nullptr ||
      bindings.resident_inputs.count != 0u ||
      bindings.sequence_tiles != nullptr ||
      bindings.sequence_tile_count != 0u || !bindings.has_staged_outputs()) {
    adapter.last_error = "accel_cpu_staged_bindings_required";
    return false;
  }
  return true;
}

[[nodiscard]] bool RunCpuMap(
    CpuAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const rund::kernel::ExecutionMetadata &metadata,
    const rund::kernel::BindingSet &bindings) {
  using rund::kernel::compute_lowering_detail::PlanIR;
  const rund::kernel::ComputeIR ir = PlanIR(plan);
  const cpu_simd_detail::CpuSimdDispatch dispatch =
      cpu_simd_detail::PrepareCpuSimdDispatch(ir, adapter.caps, input,
                                              bindings);
  if (!dispatch.prepared.ok || dispatch.run == nullptr) {
    adapter.last_error = dispatch.prepared.reason;
    return false;
  }

  static thread_local std::vector<std::max_align_t> scratch;
  const std::size_t bytes = dispatch.scratch_bytes(dispatch.prepared);
  scratch.resize((bytes + sizeof(std::max_align_t) - 1u) /
                 sizeof(std::max_align_t));
  cpu_simd_detail::CpuSimdBindingStorage binding_storage{};
  const cpu_simd_detail::CpuSimdBindingView binding_view =
      cpu_simd_detail::BindingView(bindings, binding_storage);
  const cpu_simd_detail::IndexCheck indices = cpu_simd_detail::ValidateIndices(
      binding_view, metadata.read_routes, bindings.tile_count);
  if (!indices.ok()) {
    adapter.last_error = indices.reason;
    return false;
  }
  const CpuSimdRunResult run = dispatch.run(
      dispatch.prepared,
      cpu_simd_detail::CpuSimdInvocation{
          .bindings = &binding_view,
          .count = bindings.tile_count,
      },
      cpu_simd_detail::CpuSimdScratch{
          scratch.data(), scratch.size() * sizeof(std::max_align_t)});
  if (!run.ok) {
    adapter.last_error = run.reason;
    return false;
  }
  RecordCpuDispatches(adapter, 1u);
  return true;
}

} // namespace

[[nodiscard]] bool
ExecuteCpu(void *const context, const rund::kernel::ComputePlan &plan,
           const rund::kernel::LoweringArtifact &artifact,
           const rund::kernel::ComputeDispatchWindow *const windows,
           const rund::kernel::u64 window_count,
           const rund::kernel::BindingSet &bindings) {
  auto *const adapter = static_cast<CpuAdapter *>(context);
  if (adapter == nullptr) {
    return false;
  }
  adapter->last_error = "ok";
  if (!ValidateCpuMap(*adapter, plan, artifact.metadata, windows, window_count,
                      bindings)) {
    return false;
  }

  using rund::kernel::compute_lowering_detail::AdmitArtifact;
  rund::kernel::compute_lowering_detail::ArtifactAdmission admission =
      AdmitArtifact(plan, artifact);
  if (!admission.ok) {
    adapter->last_error = admission.reason;
    return false;
  }
  return RunCpuMap(*adapter, plan, admission.input, artifact.metadata,
                   bindings);
}

bool ExecuteRetainedCpu(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const rund::kernel::ExecutionMetadata &metadata,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings) {
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->last_error = "ok";
  return ValidateCpuMap(*adapter, plan, metadata, windows, window_count,
                        bindings) &&
         RunCpuMap(*adapter, plan, input, metadata, bindings);
}

[[nodiscard]] const char *CpuLastError(void *const context) noexcept {
  const auto *const adapter = static_cast<const CpuAdapter *>(context);
  return adapter == nullptr ? "accel_cpu_backend_invalid" : adapter->last_error;
}

} // namespace rund::node::accel::detail
