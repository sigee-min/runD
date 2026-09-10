#include "internal.hpp"

#include "src/accel/kernel/fault.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency::service_free_direct_test {

bool product_unknown_quarantine(
    rund::compute::Pipeline &pipeline,
    const std::shared_ptr<rund::compute::detail::PipelineState>
        &state) noexcept {
  using namespace rund::compute;
  detail::AccelDeviceState *const native =
      state == nullptr || state->device == nullptr
          ? nullptr
          : detail::accel_device(*state->device);
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return false;
  }
  const std::uint64_t generation = pipeline.generation();
  const Status lost = pipeline.run();
  const detail::ServiceFreeDirectProductEvidence evidence =
      state->service_free_direct;
  const Status replay = pipeline.run();
  const bool valid = !lost && lost.reason() == Reason::DeviceLost &&
                     pipeline.generation() == generation && evidence.selected &&
                     evidence.quarantined &&
                     evidence.public_handoff_count == 1u &&
                     evidence.native_submit_count == 1u &&
                     evidence.epoch_native_submit_count == 0u &&
                     evidence.payload_dispatch_count == 1u &&
                     evidence.host_service_turn_count == 0u &&
                     evidence.host_epoch_callback_count == 0u &&
                     evidence.final_callback_count == 1u &&
                     evidence.authority_publication_count == 0u &&
                     state->phase == detail::PipelinePhase::Running &&
                     !replay && replay.reason() == Reason::PipelineBusy;
  if (!valid) {
    std::fprintf(
        stderr,
        "service-free Unknown lost=%u gen=%llu/%llu selected=%u quarantine=%u "
        "handoff=%llu submit=%llu epoch=%llu dispatch=%llu service=%llu "
        "callback=%llu "
        "final=%llu publication=%llu phase=%u replay=%u\n",
        static_cast<unsigned>(lost.reason()),
        static_cast<unsigned long long>(generation),
        static_cast<unsigned long long>(pipeline.generation()),
        static_cast<unsigned>(evidence.selected),
        static_cast<unsigned>(evidence.quarantined),
        static_cast<unsigned long long>(evidence.public_handoff_count),
        static_cast<unsigned long long>(evidence.native_submit_count),
        static_cast<unsigned long long>(evidence.epoch_native_submit_count),
        static_cast<unsigned long long>(evidence.payload_dispatch_count),
        static_cast<unsigned long long>(evidence.host_service_turn_count),
        static_cast<unsigned long long>(evidence.host_epoch_callback_count),
        static_cast<unsigned long long>(evidence.final_callback_count),
        static_cast<unsigned long long>(evidence.authority_publication_count),
        static_cast<unsigned>(state->phase),
        static_cast<unsigned>(replay.reason()));
  }
  return valid;
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
