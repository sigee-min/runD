#pragma once

#include <cstdint>

namespace rund::compute::detail {

// Fixed-size public-product evidence retained by PipelineState. This is the
// product routing receipt, not backend diagnostic telemetry.
struct ServiceFreeDirectProductEvidence final {
  std::uint64_t iterations{};
  std::uint64_t completed_iterations{};
  std::uint64_t public_handoff_count{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t payload_dispatch_count{};
  std::uint64_t host_service_turn_count{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t final_callback_count{};
  std::uint64_t authority_publication_count{};
  std::uint32_t registered_state_count{};
  bool selected{};
  bool fixed_native_storage{};
  bool fixed_common_storage{};
  bool quarantined{};
};

} // namespace rund::compute::detail
