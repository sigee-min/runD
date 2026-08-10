#pragma once

namespace rund::compute::detail {

struct PipelineState;

enum class PipelineClaimAuthority : unsigned char {
  Shared,
  PrivateResidency,
};

// One positive proof for the claim-free VirtualPipeline path. The proof is
// valid only for a prepared residency Pipeline whose complete canonical
// resource/claim set is Pipeline-owned. Ordinary external-resource Pipelines
// therefore cannot select the private authority.
[[nodiscard]] bool
has_private_residency_authority(const PipelineState &state) noexcept;

} // namespace rund::compute::detail
