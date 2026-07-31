#pragma once

#include <kernel/program/compute/lowering/admission.hpp>

namespace rund::kernel::compute_lowering_detail {

// The sole authenticated handoff from a public artifact to backend
// preparation. It borrows canonical/source payloads and owns only the parsed
// form required by cold CPU preparation.
struct ArtifactAdmission {
  ComputeInputAdmission input{};
  u32 emission_count = 0u;
  bool ok = false;
  const char *reason = "compute_artifact_mismatch";

  [[nodiscard]] u32 parse_count() const noexcept;
};

// A retained graph token was authenticated when minted. Warm admission checks
// only fixed-size identity and typed-owner invariants, so both counters remain
// zero by construction.
struct RetainedAdmission {
  u32 parse_count = 0u;
  u32 emission_count = 0u;
  bool ok = false;
  const char *reason = "compute_artifact_mismatch";
};

[[nodiscard]] RetainedAdmission
AdmitRetained(const ComputePlan &plan, const LoweringArtifact &artifact,
              const ComputeInputAdmission *cpu_input);

[[nodiscard]] ComputeIR PlanIR(const ComputePlan &plan) noexcept;

[[nodiscard]] ArtifactAdmission AdmitArtifact(const ComputePlan &plan,
                                              const LoweringArtifact &artifact);

[[nodiscard]] ArtifactAdmission AdmitArtifact(const ComputeIR &ir,
                                              ComputeApi api,
                                              const LoweringArtifact &artifact);

} // namespace rund::kernel::compute_lowering_detail
