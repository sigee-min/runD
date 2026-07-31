#pragma once

#include <kernel/program/compute/fusion.hpp>
#include <kernel/program/compute/metadata.hpp>

namespace rund::kernel {

struct ComputeFusedMapChainIR {
  ComputeIR ir{};
  ExecutionMetadata metadata{};
  FusionPlan fusion{};
  bool ok = false;
  const char *reason = "compute_fusion_invalid";

  [[nodiscard]] explicit operator bool() const noexcept { return ok; }
};

namespace compute_lowering_detail {

[[nodiscard]] inline ComputeFusedMapChainIR
RejectFusedMapChain(const FusionPlan &fusion, const char *const reason) {
  return ComputeFusedMapChainIR{
      .fusion = fusion,
      .reason = reason,
  };
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
