#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

void add(std::uint64_t &target, const std::uint64_t value) noexcept {
  target = value > std::numeric_limits<std::uint64_t>::max() - target
               ? std::numeric_limits<std::uint64_t>::max()
               : target + value;
}

} // namespace

void fold_stats(DeviceVsmProductRun &run) noexcept {
  if (run.stats == nullptr || run.owner == nullptr) {
    return;
  }
  const node::accel::detail::DeviceVsmEvidence &native =
      run.owner->evidence->native;
  ResidencyStats &residency = run.stats->pipeline.residency;
  const std::uint64_t frame_capacity = run.projection->frame_capacity;
  const std::uint64_t capacity_epochs =
      native.generated_epochs / frame_capacity +
      static_cast<std::uint64_t>(native.generated_epochs % frame_capacity !=
                                 0u);
  add(residency.epoch_count, capacity_epochs);
  add(residency.window_handoff_count, 1u);
  add(residency.window_batch_count, native.native_submit_count == 0u ? 0u : 1u);
  add(residency.window_queue_call_count, native.native_submit_count);
  const std::uint64_t physical_input_pages =
      run.input_count == 0u ? 0u
      : native.forecasted_pages >
              std::numeric_limits<std::uint64_t>::max() / run.input_count
          ? std::numeric_limits<std::uint64_t>::max()
          : native.forecasted_pages * run.input_count;
  add(residency.page_in_count, physical_input_pages);
  add(residency.page_out_count, native.persisted_pages);
  add(residency.page_in_bytes, native.gpu_backing_read_bytes);
  add(residency.page_out_bytes, native.gpu_backing_write_bytes);
  const std::uint64_t frame_peak = std::min<std::uint64_t>(
      native.page_count,
      frame_capacity * static_cast<std::uint64_t>(residency::Pool::BankCount));
  residency.resident_frames_peak =
      std::max(residency.resident_frames_peak, frame_peak);
  add(run.stats->command_submits, native.native_submit_count);
  const std::uint64_t dispatches = native.tile_dispatch_count != 0u
                                       ? native.tile_dispatch_count
                                       : native.payload_dispatch_count;
  add(run.stats->dispatches, dispatches);
  add(run.stats->final_dispatches, dispatches);
  add(run.stats->kernel_ns, native.kernel_ns);
  add(run.stats->kernel_samples, native.kernel_samples);
  add(run.stats->submit_wait_ns, native.submit_wait_ns);
  run.stats->command_inflight_peak =
      std::max<std::uint64_t>(run.stats->command_inflight_peak,
                              native.native_submit_count == 0u ? 0u : 1u);
}

} // namespace rund::compute::detail::device_vsm_product_detail
