#include "local.hpp"

#include "../epoch.hpp"
#include "../overlap.hpp"

#include <array>

namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunDispatchResult
dispatch_epochs(VirtualPipelineState &state, VirtualBacking &input,
                VirtualBacking &output, const VirtualRunProjection &run,
                Stats &stats, VirtualRunWork &work,
                VirtualRunTransaction *const transaction) noexcept {
  ::rund::node::hash_detail::Fnv output_hash{};
  std::array<bool, 2u> prefetch_pending{};
  if (run.scan) {
    for (std::uint64_t epoch = 0u; epoch < run.active.stream.epoch_count();
         ++epoch) {
      const VirtualEpochResult epoch_result = execute_virtual_epoch(
          state, input, output, run, epoch, prefetch_pending, stats,
          output_hash, nullptr, &work.scan, transaction);
      if (!epoch_result.status) {
        return VirtualRunDispatchResult{
            .status = epoch_result.status,
            .failed_page = epoch_result.failed_page,
            .poison_pipeline = epoch_result.poison_pipeline,
            .reduction_pending = false,
            .certainty = epoch_certainty(epoch_result.poison_pipeline),
        };
      }
    }
  } else {
    const VirtualEpochResult epoch_result =
        state.pipeline->device->backend == Backend::Cpu
            ? execute_virtual_cpu_overlap(
                  state, input, output, run, stats, output_hash,
                  run.reduction ? &work.reduction : nullptr)
            : execute_virtual_accel_overlap(
                  state, input, output, run, stats, output_hash,
                  run.reduction ? &work.reduction : nullptr);
    if (!epoch_result.status) {
      return VirtualRunDispatchResult{
          .status = epoch_result.status,
          .failed_page = epoch_result.failed_page,
          .poison_pipeline = epoch_result.poison_pipeline,
          .reduction_pending = false,
          .certainty = epoch_certainty(epoch_result.poison_pipeline),
      };
    }
  }
  return VirtualRunDispatchResult{
      .status = Status::success(),
      .output_hash = output_hash.Finish(),
      .reduction_pending = run.reduction,
  };
}

} // namespace rund::compute::detail::virtual_run_dispatch
