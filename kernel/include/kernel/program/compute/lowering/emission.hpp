#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>

#include <string>

namespace rund::kernel {

struct ComputeIR;

namespace compute_lowering_detail {

// The sole backend-emission result before a canonical IR payload is attached.
// This transient owner deliberately has no canonical byte vector: validation
// paths can authenticate an already-owned payload without copying it, while
// public artifact paths materialize the payload explicitly below.
struct ComputeArtifactEmission {
  ArtifactKey key{};
  LoweringArtifactKind kind = LoweringArtifactKind::None;
  ExecutionMetadata metadata{};
  std::string source_text{};
  u64 source_text_upper_bytes = 0u;
  u32 emission_count = 0u;
  bool ok = false;
  const char *reason = "compute_lowering_invalid";
};

// Product graph compilation keeps the admitted typed form instead of a
// second serialized canonical owner. CPU consumes the parsed form; source
// backends discard it when the graph token is minted.
struct RetainedComputeArtifact {
  LoweringArtifact artifact{};
  ComputeInputAdmission input{};
  u32 emission_count = 0u;

  [[nodiscard]] u32 parse_count() const noexcept { return input.parse_count; }
};

[[nodiscard]] ComputeArtifactEmission
RejectComputeArtifactEmission(const ArtifactKey &key, const char *reason,
                              u32 emission_count = 0u);

[[nodiscard]] ComputeArtifactEmission
EmitComputeArtifactTransient(const ComputeInputAdmission &input,
                             ExecutionMetadata metadata);

[[nodiscard]] ComputeArtifactEmission
EmitComputeArtifactTransient(const ComputeIR &ir,
                             const ComputeInputAdmission &input);

[[nodiscard]] RetainedComputeArtifact
LowerRetainedComputeArtifact(const ComputeIR &ir, ComputeApi api);

[[nodiscard]] RetainedComputeArtifact EmitAdmittedRetainedComputeArtifact(
    ExecutionMetadata metadata, ComputeInputAdmission input);

[[nodiscard]] RetainedComputeArtifact EmitGeneratedRetainedComputeArtifact(
    ComputeIR &&ir, ExecutionMetadata &&metadata, ComputeInputAdmission input);

} // namespace compute_lowering_detail
} // namespace rund::kernel
