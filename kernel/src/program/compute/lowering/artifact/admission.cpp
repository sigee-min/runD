#include <kernel/program/compute/lowering/artifact/admission.hpp>

#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/plan.hpp>

#include <string_view>
#include <utility>

namespace rund::kernel::compute_lowering_detail {

namespace {

[[nodiscard]] constexpr LoweringArtifactKind
KindFor(const ComputeApi api) noexcept {
  switch (api) {
  case ComputeApi::Metal:
    return LoweringArtifactKind::MetalSource;
  case ComputeApi::Vulkan:
    return LoweringArtifactKind::VulkanSource;
  case ComputeApi::Cpu:
    return LoweringArtifactKind::CpuPlan;
  }
  return LoweringArtifactKind::None;
}

[[nodiscard]] bool SameMap(const ComputeMap &left,
                           const ComputeMap &right) noexcept {
  return left.op_hash_hi == right.op_hash_hi &&
         left.op_hash_lo == right.op_hash_lo && left.api == right.api &&
         left.scalar == right.scalar && left.domain == right.domain &&
         left.fixed_format == right.fixed_format &&
         left.input_buffer_count == right.input_buffer_count &&
         left.output_buffer_count == right.output_buffer_count &&
         left.input_bytes_per_tile == right.input_bytes_per_tile &&
         left.output_bytes_per_tile == right.output_bytes_per_tile &&
         left.param_bytes == right.param_bytes &&
         left.metadata_bytes_per_tile == right.metadata_bytes_per_tile;
}

[[nodiscard]] bool SameReason(const char *const left,
                              const char *const right) noexcept {
  if (left == nullptr || right == nullptr) {
    return left == right;
  }
  return std::string_view{left} == std::string_view{right};
}

[[nodiscard]] bool SameMetadata(const ExecutionMetadata &left,
                                const ExecutionMetadata &right) noexcept {
  return SameMap(left.map, right.map) &&
         left.param_storage == right.param_storage &&
         left.input_element_bytes == right.input_element_bytes &&
         left.output_element_bytes == right.output_element_bytes &&
         left.binding_accesses == right.binding_accesses &&
         left.binding_names == right.binding_names &&
         left.read_routes == right.read_routes &&
         left.direct_read_mask == right.direct_read_mask &&
         left.uniform_read_mask == right.uniform_read_mask &&
         left.read_count == right.read_count &&
         left.write_count == right.write_count &&
         left.uses_index == right.uses_index && left.ok == right.ok &&
         SameReason(left.reason, right.reason);
}

[[nodiscard]] ArtifactAdmission Reject(ComputeInputAdmission input,
                                       const char *const reason,
                                       const u32 emission_count = 0u) {
  return ArtifactAdmission{
      .input = std::move(input),
      .emission_count = emission_count,
      .reason = reason,
  };
}

[[nodiscard]] ArtifactAdmission Admit(const ComputeIR &ir, const ComputeApi api,
                                      const LoweringArtifact &artifact,
                                      const ComputePlan *const plan) {
  const ArtifactKey expected_key =
      MakeKey(ir, api, artifact.canonical_ir_bytes);

  // Authenticate fixed-size identity before parsing or source emission.
  if (!artifact.ok || artifact.kind != KindFor(api) ||
      artifact.canonical_ir_bytes.empty() || artifact.source_text.empty()) {
    return Reject(ComputeInputAdmission{.key = expected_key},
                  artifact.kind == KindFor(api)
                      ? "compute_artifact_mismatch"
                      : "compute_artifact_non_executable");
  }
  if (expected_key.canonical_ir_hash_hi != ir.op_hash_hi ||
      expected_key.canonical_ir_hash_lo != ir.op_hash_lo) {
    return Reject(ComputeInputAdmission{.key = expected_key},
                  "compute_ir_hash_mismatch");
  }
  if (artifact.key != expected_key) {
    return Reject(ComputeInputAdmission{.key = expected_key},
                  "compute_artifact_mismatch");
  }

  ComputeInputAdmission input = AdmitComputeInputWithKey(
      ir, api, artifact.canonical_ir_bytes, expected_key);
  if (!input.ok) {
    const char *const reason = input.reason;
    return Reject(std::move(input), reason);
  }

  ComputeArtifactEmission emission = EmitComputeArtifactTransient(ir, input);
  const u32 emission_count = emission.emission_count;
  if (!emission.ok) {
    const char *const reason = emission.reason;
    return Reject(std::move(input), reason, emission_count);
  }

  if (emission.key != artifact.key || emission.kind != artifact.kind ||
      (plan != nullptr &&
       !ComputePlanMatchesMap(*plan, emission.metadata.map)) ||
      !SameMetadata(emission.metadata, artifact.metadata) ||
      emission.source_text != artifact.source_text) {
    return Reject(std::move(input), "compute_artifact_mismatch",
                  emission_count);
  }

  return ArtifactAdmission{
      .input = std::move(input),
      .emission_count = emission_count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace

u32 ArtifactAdmission::parse_count() const noexcept {
  return input.parse_count;
}

RetainedAdmission AdmitRetained(const ComputePlan &plan,
                                const LoweringArtifact &artifact,
                                const ComputeInputAdmission *const cpu_input) {
  const bool cpu = plan.api == ComputeApi::Cpu;
  const ArtifactKey &key = artifact.key;
  const bool key_matches =
      key.api == plan.api && key.scalar == plan.scalar &&
      key.domain == plan.domain && key.fixed_format == plan.fixed_format &&
      key.op_hash_hi == plan.op_hash_hi && key.op_hash_lo == plan.op_hash_lo &&
      key.canonical_ir_hash_hi == plan.op_hash_hi &&
      key.canonical_ir_hash_lo == plan.op_hash_lo;
  const bool artifact_matches =
      artifact.ok && SameReason(artifact.reason, "ok") &&
      artifact.kind == KindFor(plan.api) && key_matches &&
      artifact.canonical_ir_bytes.empty() && artifact.metadata.ok &&
      SameReason(artifact.metadata.reason, "ok") &&
      ComputePlanMatchesMap(plan, artifact.metadata.map) &&
      (cpu ? artifact.source_text.empty() : !artifact.source_text.empty());
  if (!artifact_matches) {
    return {};
  }
  if (!cpu) {
    return cpu_input == nullptr ? RetainedAdmission{.ok = true, .reason = "ok"}
                                : RetainedAdmission{};
  }
  if (cpu_input == nullptr || !cpu_input->ok ||
      !SameReason(cpu_input->reason, "ok") || cpu_input->key != artifact.key ||
      !cpu_input->parsed.ok || !SameReason(cpu_input->parsed.reason, "ok") ||
      cpu_input->parsed.scalar_mode !=
          DomainModeFor(plan.scalar, plan.domain) ||
      cpu_input->parsed.fixed_format != plan.fixed_format) {
    return {};
  }
  return RetainedAdmission{.ok = true, .reason = "ok"};
}

ComputeIR PlanIR(const ComputePlan &plan) noexcept {
  return ComputeIR{
      .scalar = plan.scalar,
      .domain = plan.domain,
      .fixed_format = plan.fixed_format,
      .op_hash_hi = plan.op_hash_hi,
      .op_hash_lo = plan.op_hash_lo,
      .canonical_bytes = {},
      .ok = true,
      .reason = "ok",
  };
}

ArtifactAdmission AdmitArtifact(const ComputePlan &plan,
                                const LoweringArtifact &artifact) {
  const ComputeIR ir = PlanIR(plan);
  return Admit(ir, plan.api, artifact, &plan);
}

ArtifactAdmission AdmitArtifact(const ComputeIR &ir, const ComputeApi api,
                                const LoweringArtifact &artifact) {
  if (artifact.canonical_ir_bytes != ir.canonical_bytes) {
    return Reject(
        ComputeInputAdmission{
            .key = MakeKey(ir, api, artifact.canonical_ir_bytes)},
        "compute_artifact_mismatch");
  }
  return Admit(ir, api, artifact, nullptr);
}

} // namespace rund::kernel::compute_lowering_detail
