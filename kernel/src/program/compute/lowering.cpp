#include <kernel/program/compute/lowering/entry.hpp>

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/cpu.hpp>
#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/lowering/metadata.hpp>
#include <kernel/program/compute/lowering/metal/source.hpp>
#include <kernel/program/compute/lowering/vulkan.hpp>

#include <utility>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] ComputeArtifactEmission
RejectComputeArtifactEmission(const ArtifactKey &key, const char *const reason,
                              const u32 emission_count) {
  return ComputeArtifactEmission{
      .key = key,
      .metadata = ExecutionMetadata{.reason = reason},
      .emission_count = emission_count,
      .reason = reason,
  };
}

[[nodiscard]] ComputeArtifactEmission
EmitComputeArtifactTransient(const ComputeInputAdmission &input,
                             ExecutionMetadata metadata) {
  if (!input.ok) {
    return RejectComputeArtifactEmission(input.key, input.reason);
  }
  if (!metadata.ok) {
    return RejectComputeArtifactEmission(input.key, metadata.reason, 1u);
  }

  const std::vector<BindingLayout> layouts = BuildBindingLayout(input.parsed);

  std::string text;
  LoweringArtifactKind kind = LoweringArtifactKind::None;
  if (input.key.api == ComputeApi::Metal) {
    text = MetalSource(input.parsed, input.key, layouts);
    kind = LoweringArtifactKind::MetalSource;
  } else if (input.key.api == ComputeApi::Vulkan) {
    text = VulkanSource(input.parsed, input.key, layouts);
    kind = LoweringArtifactKind::VulkanSource;
  } else if (input.key.api == ComputeApi::Cpu) {
    text = CpuPlanText(input.parsed, input.key, layouts);
    kind = LoweringArtifactKind::CpuPlan;
  } else {
    return RejectComputeArtifactEmission(input.key, "compute_lowering_invalid",
                                         1u);
  }

  return ComputeArtifactEmission{
      .key = input.key,
      .kind = kind,
      .metadata = std::move(metadata),
      .source_text = std::move(text),
      .emission_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] ComputeArtifactEmission
EmitComputeArtifactTransient(const ComputeIR &ir,
                             const ComputeInputAdmission &input) {
  if (!input.ok) {
    return RejectComputeArtifactEmission(input.key, input.reason);
  }
  return EmitComputeArtifactTransient(
      input, MetadataFromParsed(ir, input.key.api, input.parsed));
}

namespace {

[[nodiscard]] LoweringArtifact
MaterializeComputeArtifact(ComputeArtifactEmission emission,
                           std::vector<u8> canonical_bytes = {}) {
  return LoweringArtifact{
      .key = emission.key,
      .kind = emission.kind,
      .metadata = std::move(emission.metadata),
      .source_text = std::move(emission.source_text),
      .canonical_ir_bytes = std::move(canonical_bytes),
      .ok = emission.ok,
      .reason = emission.reason,
  };
}

[[nodiscard]] LoweringArtifact
EmitComputeArtifact(const ComputeIR &ir, const std::vector<u8> &canonical_bytes,
                    const ComputeInputAdmission &input) {
  ComputeArtifactEmission emission = EmitComputeArtifactTransient(ir, input);
  if (!emission.ok) {
    return MaterializeComputeArtifact(std::move(emission));
  }
  return MaterializeComputeArtifact(
      std::move(emission),
      std::vector<u8>(canonical_bytes.begin(), canonical_bytes.end()));
}

} // namespace

[[nodiscard]] RetainedComputeArtifact
LowerRetainedComputeArtifact(const ComputeIR &ir, const ComputeApi api) {
  ComputeInputAdmission input = AdmitComputeInput(ir, api);
  ExecutionMetadata metadata =
      input.ok ? MetadataFromParsed(ir, api, input.parsed)
               : ExecutionMetadata{.reason = input.reason};
  return EmitAdmittedRetainedComputeArtifact(std::move(metadata),
                                             std::move(input));
}

[[nodiscard]] RetainedComputeArtifact EmitAdmittedRetainedComputeArtifact(
    ExecutionMetadata metadata, ComputeInputAdmission input) {
  ComputeArtifactEmission emission =
      EmitComputeArtifactTransient(input, std::move(metadata));
  const u32 emission_count = emission.emission_count;
  return RetainedComputeArtifact{
      .artifact = MaterializeComputeArtifact(std::move(emission)),
      .input = std::move(input),
      .emission_count = emission_count,
  };
}

[[nodiscard]] RetainedComputeArtifact EmitGeneratedRetainedComputeArtifact(
    ComputeIR &&ir, ExecutionMetadata &&metadata, ComputeInputAdmission input) {
  // The private graph token is authenticated by its sealed control-block
  // capability, typed admission, and key. Release generated canonical storage
  // at the handoff; no retained consumer can authenticate or execute from
  // those bytes.
  std::vector<u8>{}.swap(ir.canonical_bytes);
  return EmitAdmittedRetainedComputeArtifact(std::move(metadata),
                                             std::move(input));
}

} // namespace compute_lowering_detail

[[nodiscard]] LoweringArtifact LowerComputeIR(const ComputeIR &ir,
                                              const ComputeApi api) {
  const compute_lowering_detail::ComputeInputAdmission input =
      compute_lowering_detail::AdmitComputeInput(ir, api);
  return compute_lowering_detail::EmitComputeArtifact(ir, ir.canonical_bytes,
                                                      input);
}

} // namespace rund::kernel
