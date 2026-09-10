#include "internal.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace rund::node::accel::detail::metal_spatial_window_proof_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] const char *
DiagnosticName(const DiagnosticReason reason) noexcept {
  switch (reason) {
  case DiagnosticReason::Recurrence:
    return "recurrence";
  case DiagnosticReason::ExecutionShape:
    return "execution_shape";
  case DiagnosticReason::EntryNull:
    return "entry_null";
  case DiagnosticReason::EntrySemanticMismatch:
    return "entry_semantic_mismatch";
  case DiagnosticReason::DeclaredRange:
    return "declared_range";
  case DiagnosticReason::TopologyOperator:
    return "topology_operator";
  case DiagnosticReason::SharedHaloCandidate:
    return "shared_halo_candidate";
  case DiagnosticReason::SharedHaloIdentity:
    return "shared_halo_identity";
  case DiagnosticReason::BindingCapture:
    return "binding_capture";
  case DiagnosticReason::BindingAlias:
    return "binding_alias";
  case DiagnosticReason::BindingHandle:
    return "binding_handle";
  case DiagnosticReason::BindingOwner:
    return "binding_owner";
  }
  return "unknown";
}

} // namespace

void RecordDiagnostic(const DiagnosticReason reason, const std::uint64_t value0,
                      const std::uint64_t value1, const std::uint64_t value2,
                      const std::uint64_t value3) noexcept {
  const char *const enabled =
      std::getenv("RUND_METAL_SPATIAL_WINDOW_DIAGNOSTIC");
  if (enabled == nullptr || std::string_view{enabled} != "1") {
    return;
  }
  static std::atomic<bool> emitted{false};
  if (!emitted.exchange(true, std::memory_order_acq_rel)) {
    std::fprintf(
        stderr,
        "Metal spatial proof first_false=%s values=%llu,%llu,%llu,%llu\n",
        DiagnosticName(reason), static_cast<unsigned long long>(value0),
        static_cast<unsigned long long>(value1),
        static_cast<unsigned long long>(value2),
        static_cast<unsigned long long>(value3));
  }
}

#endif

} // namespace rund::node::accel::detail::metal_spatial_window_proof_internal
