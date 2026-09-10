#include "internal.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>

namespace rund_node_test_virtual::product::graph_pointwise {

std::array<std::uint64_t, StageCount>
stage_generations(const Case &test_case) noexcept {
  std::array<std::uint64_t, StageCount> generations{};
  if (test_case.state == nullptr ||
      test_case.state->graph_pipelines.size() !=
          StageCount * rund::compute::detail::residency::Pool::BankCount) {
    generations.fill(std::numeric_limits<std::uint64_t>::max());
    return generations;
  }
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const auto &pipeline =
        test_case.state->graph_pipelines
            [stage * rund::compute::detail::residency::Pool::BankCount];
    if (pipeline == nullptr || pipeline->publication == nullptr) {
      generations.fill(std::numeric_limits<std::uint64_t>::max());
      return generations;
    }
    std::lock_guard state_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    generations[stage] = pipeline->publication->generation;
  }
  return generations;
}

bool observe_output(Case &test_case) noexcept {
  std::vector<std::uint64_t> observed(test_case.expected.size());
  return test_case.output_backing->observe(
             std::as_writable_bytes(std::span{observed})) &&
         std::equal(observed.begin(), observed.end(),
                    test_case.expected.begin(), test_case.expected.end());
}

bool validate_success(Case &test_case, const rund::compute::Backend backend,
                      const rund::compute::Status &status,
                      const std::uint64_t initial_version) noexcept {
  using namespace rund::compute;
  const Stats stats = test_case.pipeline.stats();
  const ResidencyStats residency = stats.pipeline.residency;
  const BackingFacts input = test_case.input_backing->facts();
  const BackingFacts output = test_case.output_backing->facts();
  const std::uint64_t frame_capacity =
      test_case.pipeline.plan().residency.frame_capacity;
  const std::uint64_t batches =
      frame_capacity == 0u
          ? 0u
          : (test_case.page_count + frame_capacity - 1u) / frame_capacity;
  const auto owner =
      test_case.state == nullptr
          ? std::shared_ptr<
                detail::device_vsm_product_detail::DeviceVsmProductOwner>{}
          : std::static_pointer_cast<
                detail::device_vsm_product_detail::DeviceVsmProductOwner>(
                test_case.state->device_vsm_product_cache);
  const auto *const evidence = owner == nullptr || owner->evidence == nullptr
                                   ? nullptr
                                   : owner->evidence.get();
  std::uint32_t expected_reuse = 0u;
  std::uint32_t expected_checksum = 0u;
  std::uint32_t expected_round_trips = 0u;
  std::uint64_t expected_state_bytes = 0u;
  std::uint64_t expected_scratch_bytes = 0u;
  const bool exact_ring =
      owner != nullptr && owner->proof != nullptr &&
      rund::node::accel::detail::device_vsm_ring_schedule_expected(
          owner->proof->geometry, owner->proof->width, expected_reuse,
          expected_checksum) &&
      rund::node::accel::detail::device_vsm_ring_storage_expected(
          owner->proof->geometry, owner->proof->width,
          owner->proof->residents.count, expected_state_bytes,
          expected_scratch_bytes) &&
      rund::node::accel::detail::device_vsm_ring_round_trips_expected(
          owner->proof->geometry, owner->proof->residents.count,
          expected_round_trips);
  const bool exact_native =
      exact_ring && evidence != nullptr &&
      evidence->public_handoff_count == 1u &&
      evidence->authority_accept_count == 1u &&
      evidence->pipeline_terminal_count == StageCount &&
      evidence->backing_publication_count == 1u && evidence->final_received &&
      !evidence->quarantined &&
      evidence->native.page_count == test_case.page_count &&
      evidence->native.generated_epochs == test_case.page_count &&
      evidence->native.completed_epochs == test_case.page_count &&
      evidence->native.ring_reuse_transitions == expected_reuse &&
      evidence->native.ring_schedule_checksum == expected_checksum &&
      evidence->native.ring_round_trips == expected_round_trips &&
      evidence->native.ring_state_bytes == expected_state_bytes &&
      evidence->native.ring_scratch_bytes == expected_scratch_bytes &&
      evidence->native.native_submit_count == 1u &&
      evidence->native.epoch_native_submit_count == 0u &&
      evidence->native.payload_dispatch_count == 1u &&
      evidence->native.host_service_turn_count == 0u &&
      evidence->native.host_epoch_callback_count == 0u &&
      evidence->native.final_callback_count == 1u;
  const bool device_vsm = backend != Backend::Cpu && exact_native &&
                          residency.window_handoff_count == 1u &&
                          residency.window_batch_count == 1u &&
                          residency.window_queue_call_count == 1u;
  const std::uint64_t expected_epochs =
      device_vsm ? batches : batches * StageCount;
  const std::size_t logical_bytes =
      test_case.expected.size() * sizeof(std::uint64_t);
  const bool valid =
      status && frame_capacity == 2u &&
      (backend == Backend::Cpu || device_vsm) &&
      residency.epoch_count == expected_epochs &&
      residency.page_in_count == test_case.page_count &&
      residency.backing_read_bytes == logical_bytes &&
      residency.page_out_count == test_case.page_count &&
      residency.backing_write_bytes == logical_bytes &&
      input.read_count == test_case.page_count &&
      input.read_bytes == logical_bytes &&
      output.write_count == test_case.page_count &&
      output.write_bytes == logical_bytes &&
      detail::VirtualBackingAccess::version(*test_case.output_backing) ==
          initial_version + 1u &&
      detail::VirtualBackingAccess::recovery_bytes(*test_case.output_backing) ==
          0u &&
      observe_output(test_case) && test_case.output_backing->tail_poisoned() &&
      (!device_vsm || (stats.command_submits == 1u && stats.dispatches == 1u &&
                       stats.final_dispatches == 1u));
  if (!valid) {
    std::fprintf(
        stderr,
        "Graph pointwise backend=%u pages=%zu reason=%.*s vsm=%u k=%llu "
        "epochs=%llu/%llu submit=%llu dispatch=%llu in=%llu/%llu "
        "out=%llu/%llu version=%llu\n",
        static_cast<unsigned>(backend), test_case.page_count,
        static_cast<int>(status.error().size()), status.error().data(),
        static_cast<unsigned>(device_vsm),
        static_cast<unsigned long long>(frame_capacity),
        static_cast<unsigned long long>(residency.epoch_count),
        static_cast<unsigned long long>(expected_epochs),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(input.read_count),
        static_cast<unsigned long long>(input.read_bytes),
        static_cast<unsigned long long>(output.write_count),
        static_cast<unsigned long long>(output.write_bytes),
        static_cast<unsigned long long>(
            detail::VirtualBackingAccess::version(*test_case.output_backing)));
  }
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise
