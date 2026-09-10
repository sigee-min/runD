#pragma once

#include "src/accel/kernel/residency/persistent_sliding.hpp"

#include <cstdint>

namespace rund_node_test_pipeline_residency::persistent_sliding_test {

namespace accel = rund::node::accel::detail;

struct FakePersistentSlidingState final {
  accel::PersistentResidencySlidingCapability capability{};
  accel::PersistentResidencySlidingRequest request{};
  std::uint64_t submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t backend_epoch_callback_count{};
  std::uint64_t gpu_completed_coordinates{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t final_count{};
  bool final_sent{};
};

[[nodiscard]] accel::PersistentResidencySlidingSubmitResult
FakePersistentSlidingSubmit(
    const accel::PersistentResidencySlidingRequest &,
    accel::PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] rund::AccelCheck FakePersistentSlidingWaitDone(
    accel::PersistentResidencySlidingControl &,
    const accel::PersistentResidencySlidingDoneWait &,
    accel::PersistentResidencySlidingDoneObservation &) noexcept;
[[nodiscard]] rund::AccelCheck FakePersistentSlidingSignalReady(
    accel::PersistentResidencySlidingControl &,
    const accel::PersistentResidencySlidingReadySignal &) noexcept;
[[nodiscard]] rund::AccelCheck FakePersistentSlidingAckDone(
    accel::PersistentResidencySlidingControl &,
    const accel::PersistentResidencySlidingAcknowledgeDone &) noexcept;
[[nodiscard]] rund::AccelCheck FakePersistentSlidingFailService(
    accel::PersistentResidencySlidingControl &,
    const accel::PersistentResidencySlidingServiceFailure &) noexcept;

inline constexpr accel::PersistentResidencySlidingServiceOps FakeServiceOps{
    .submit_result = FakePersistentSlidingSubmit,
    .wait_done = FakePersistentSlidingWaitDone,
    .signal_ready = FakePersistentSlidingSignalReady,
    .ack_done = FakePersistentSlidingAckDone,
    .fail_service = FakePersistentSlidingFailService,
};

} // namespace rund_node_test_pipeline_residency::persistent_sliding_test
