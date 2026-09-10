#pragma once

#include "proof.hpp"

#include "../../prepared/interface/api.hpp"

#include <accel/check.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

enum class DeviceVsmProjectionKind : std::uint8_t {
  Direct,
  GraphPointwise,
  GraphMapReduce,
  Scan,
  Reduce,
  GraphResident,
  WindowRing,
};

struct DeviceVsmProofRequest final {
  PreparedKernelPipeline pipeline{};
  // Optional second fixed-W bank. Direct requires the same canonical Map.
  // GraphPointwise instead binds the exact terminal Map stage whose sole Read
  // consumes the primary stage's sole output.
  PreparedKernelPipeline peer_pipeline{};
  // Exact collective authority for GraphMapReduce. It is distinct from the
  // optional same-Map peer bank above and must contain one canonical Reduce.
  PreparedKernelPipeline graph_collective_pipeline{};
  // Complete stage-ordered bank-0 authority for a fused Graph pointwise
  // chain. Every stage remains strongly owned by the resulting proof; a
  // first/terminal-only projection would leave intermediate generations and
  // semantics outside the one-run transaction.
  std::array<PreparedKernelPipeline, DeviceVsmGraphStageCapacity>
      graph_pointwise_pipelines{};
  std::uint8_t graph_pointwise_pipeline_count{};
  DeviceVsmGraphPointwiseTopology graph_pointwise_topology{};
  DeviceVsmPageMap graph_pointwise_page_map{};
  DeviceVsmGraphResidentProof graph_resident{};
  std::array<PreparedKernelPipeline, DeviceVsmGraphStageCapacity>
      graph_resident_pipelines{};
  std::uint8_t graph_resident_pipeline_count{};
  DeviceVsmIdentity identity{};
  DeviceVsmResidentSet residents{};
  DeviceVsmPageGeometry geometry{};
  std::uint64_t output_bytes{};
  DeviceVsmGraphWavefrontProof graph_wavefront{};
  std::uint8_t width{};
  DeviceVsmProjectionKind kind{DeviceVsmProjectionKind::Direct};
};

struct DeviceVsmProofProjection final {
  rund::AccelCheck check{};
  std::shared_ptr<const DeviceVsmProof> proof{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return check.ok && proof != nullptr;
  }
};

// Shape-only WindowRing capability. Both prepared banks are inspected through
// the canonical Window authority and fusion validators; no native or owner
// state is created or changed.
[[nodiscard]] rund::AccelCheck
QueryPreparedKernelPipelineDeviceVsmWindowCapability(
    const PreparedKernelPipeline &, const PreparedKernelPipeline &,
    rund::kernel::ComputeScalar, rund::kernel::ComputeDomain,
    rund::kernel::WindowElement) noexcept;

// Projects one already prepared pointwise Map authority into the distinct
// page-coordinate recurrence proof. The prepared Pipeline remains the strong
// semantic owner; this function never infers semantics from backend source.
[[nodiscard]] DeviceVsmProofProjection
ProjectPreparedKernelPipelineDeviceVsm(const DeviceVsmProofRequest &) noexcept;

[[nodiscard]] DeviceVsmPreparation
PrepareDeviceVsm(const rund::AccelDevice &,
                 const std::shared_ptr<const DeviceVsmProof> &) noexcept;

} // namespace rund::node::accel::detail
