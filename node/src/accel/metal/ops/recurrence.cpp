#include "internal.hpp"

#include "../kernel.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_ops_detail {
namespace {

[[nodiscard]] bool
ValidHistoryProof(const ServiceFreeDirectProof &proof) noexcept {
  if (proof.retention != ServiceFreeDirectRetention::History ||
      proof.iterations < 2u || proof.output_count == 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < proof.output_count; ++index) {
    const rund::kernel::ResidentBufferRef &ref = proof.outputs[index];
    if (proof.output_handles[index] == nullptr || ref.id == 0u ||
        ref.bytes == 0u || ref.offset_bytes > ref.bytes || ref.count == 0u ||
        ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
        ref.usage != rund::kernel::kResidentUsageWrite ||
        ref.count % proof.iterations != 0u) {
      return false;
    }
    const std::uint64_t slice_count = ref.count / proof.iterations;
    if (slice_count == 0u ||
        slice_count >
            std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
        proof.output_pitch_bytes[index] != slice_count * ref.stride_bytes) {
      return false;
    }
    const std::uint64_t last = ref.count - 1u;
    if (last > std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
        ref.offset_bytes > std::numeric_limits<std::uint64_t>::max() -
                               last * ref.stride_bytes ||
        ref.element_bytes > std::numeric_limits<std::uint64_t>::max() -
                                ref.offset_bytes - last * ref.stride_bytes ||
        ref.offset_bytes + last * ref.stride_bytes + ref.element_bytes >
            ref.bytes) {
      return false;
    }
  }
  return true;
}

} // namespace

ServiceFreeDirectCapability
ServiceFreeDirectCapabilityFor(const std::shared_ptr<void> &prepared,
                               const ServiceFreeDirectProof &proof) noexcept {
  MetalFusedDirectRecurrenceDiagnostics diagnostics{};
  const bool structural =
      proof.artifact != nullptr &&
      proof.artifact->key.api == rund::kernel::ComputeApi::Metal &&
      InspectMetalFusedDirectRecurrence(prepared, diagnostics) &&
      diagnostics.iteration_count == proof.iterations &&
      diagnostics.native_command_count != 0u &&
      diagnostics.native_dispatch_count == 1u &&
      diagnostics.native_storage_bytes != 0u &&
      diagnostics.route_host_bytes != 0u &&
      diagnostics.iteration_argument_bytes == sizeof(std::uint32_t);
  const MetalFusedDirectRecurrenceRetention expected_retention =
      proof.retention == ServiceFreeDirectRetention::History
          ? MetalFusedDirectRecurrenceRetention::History
      : proof.retention == ServiceFreeDirectRetention::Terminal
          ? MetalFusedDirectRecurrenceRetention::Terminal
          : MetalFusedDirectRecurrenceRetention::Unknown;
  const bool mode_matches =
      structural && diagnostics.retention == expected_retention;
  const bool terminal =
      mode_matches && proof.retention == ServiceFreeDirectRetention::Terminal;
  const bool history = mode_matches && ValidHistoryProof(proof);
  const bool exact = terminal || history;
  return ServiceFreeDirectCapability{
      .check = {exact, exact ? "ok" : "accel_kernel_pipeline_invalid"},
      .retained_bytes = exact ? diagnostics.retained_bytes : 0u,
      .device_generated_recurrence = exact,
      .fixed_native_storage = exact,
      .fixed_common_storage = exact && proof.fixed_common_storage,
      .one_native_submit = exact,
      .host_service_turns_zero = exact,
      .host_epoch_callbacks_zero = exact,
      .aggregate_terminal_once = exact,
      .terminal_output = terminal,
      .history_output = history,
  };
}

} // namespace rund::node::accel::detail::metal_ops_detail
