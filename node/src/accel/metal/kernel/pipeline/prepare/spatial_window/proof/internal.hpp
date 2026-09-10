#pragma once

#include "../proof.hpp"

#include <cstddef>

namespace rund::node::accel::detail::metal_spatial_window_proof_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

enum class DiagnosticReason : std::uint32_t {
  Recurrence = 1u,
  ExecutionShape = 2u,
  EntryNull = 3u,
  EntrySemanticMismatch = 4u,
  DeclaredRange = 5u,
  TopologyOperator = 6u,
  SharedHaloCandidate = 7u,
  SharedHaloIdentity = 8u,
  BindingCapture = 9u,
  BindingAlias = 10u,
  BindingHandle = 11u,
  BindingOwner = 12u,
};

void RecordDiagnostic(DiagnosticReason reason, std::uint64_t value0,
                      std::uint64_t value1, std::uint64_t value2,
                      std::uint64_t value3) noexcept;

[[nodiscard]] bool CaptureGeometry(const BackendBatchEntry &entry,
                                   const KernelExecution &execution,
                                   MetalSpatialWindowProof &proof,
                                   bool &shape_set,
                                   bool &graph_binding_set) noexcept;

[[nodiscard]] bool CaptureBindings(const BackendBatchEntry &entry,
                                   MetalSpatialWindowProof &proof,
                                   bool &binding_set) noexcept;

#endif

} // namespace rund::node::accel::detail::metal_spatial_window_proof_internal
