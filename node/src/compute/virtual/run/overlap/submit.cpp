#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../graph/reduce/timeline.hpp"
#include "../cache.hpp"
#include "../reduce.hpp"

#include <rund/compute/pipeline/runtime.hpp>

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace rund::compute::detail::virtual_run_overlap {

[[nodiscard]] Status submit_epoch(PreparedEpoch &prepared) noexcept {
  if (prepared.pipeline == nullptr ||
      prepared.pipeline->residency_pool == nullptr || prepared.submitted ||
      prepared.pipeline->residency_bank >= residency::Pool::BankCount ||
      !prepared.pipeline->residency_pool->submit_execution(
          prepared.pipeline->residency_bank, prepared.pipeline,
          execution_lease(prepared))) {
    return Status::fail(Reason::PipelineBusy);
  }
  prepared.submitted = true;
  return Status::success();
}

template <bool UseCycle>
[[nodiscard]] bool
complete_execution(PreparedEpoch &prepared, residency::Authority &authority,
                   const std::uint64_t cycle, const bool success,
                   const bool invalidate_cycle = false,
                   const bool invalidate_direct = false) noexcept {
  if constexpr (UseCycle) {
    return cycle != 0u && authority.cycles().complete_cycle(
                              cycle, prepared.token, success, invalidate_cycle);
  }
  return authority.complete(prepared.token, success, invalidate_direct);
}

template <bool UseAccelerator, bool UseCycle>
[[nodiscard]] Status
finish_epoch(PreparedEpoch &prepared, const VirtualRunProjection &run,
             Stats &stats, const std::uint64_t cycle, bool &poison) noexcept {
  residency::Pool &pool = *prepared.pipeline->residency_pool;
  const auto begin_device_drain = [&]() noexcept {
    const std::span<const residency::CacheKey> keys{prepared.keys.data(),
                                                    prepared.binding_count};
    if constexpr (UseAccelerator) {
      if (!prepared.coherent_output) {
        return pool.authority().begin_migration(keys, pool.first_output_frame,
                                                pool.output_frame_count);
      }
    }
    return run.reduction
               ? pool.authority().begin_discard(keys, pool.first_output_frame,
                                                pool.output_frame_count)
               : pool.authority().begin_writeback(keys, pool.first_output_frame,
                                                  pool.output_frame_count);
  };
  const auto reserve_drain = [&]() noexcept -> Status {
    const residency::AuthorityResult writeback = begin_device_drain();
    if (!writeback) {
      const residency::AuthorityResult recovery = begin_device_drain();
      const bool clean =
          recovery && pool.authority().discard(recovery.lease.token);
      poison = true;
      return Status::fail(clean ? Reason::PipelineInvalid
                                : Reason::PipelineBusy);
    }
    if (writeback.lease.transitions.size() > prepared.drains.size()) {
      const bool discarded = pool.authority().discard(writeback.lease.token);
      poison = true;
      return Status::fail(discarded ? Reason::PipelineInvalid
                                    : Reason::PipelineBusy);
    }
    prepared.drain_token = writeback.lease.token;
    prepared.drain_count = writeback.lease.transitions.size();
    std::copy(writeback.lease.transitions.begin(),
              writeback.lease.transitions.end(), prepared.drains.begin());
    return Status::success();
  };
  const residency::ExecutionReceipt receipt =
      pool.wait_execution(prepared.pipeline->residency_bank);
  prepared.submitted = false;
  graph_reduce::observe_timeline(prepared.timeline, receipt,
                                 stats.pipeline.residency);
  if (!receipt.status) {
    const Status folded = fold_epoch(prepared, stats);
    const bool rolled_back = complete_execution<UseCycle>(
        prepared, pool.authority(), cycle, false, true, false);
    prepared.token = 0u;
    poison = poisoned_pipeline(prepared.pipeline) || !rolled_back || !folded;
    return folded ? receipt.status : folded;
  }
  using ::rund::detail::counter::Accumulate;
  Accumulate(stats.pipeline.residency.epoch_count, 1u);
  const Status retained = retain_residency_output(
      *prepared.pipeline, run, execution_output_lease(prepared), stats);
  if (!retained) {
    const Status folded = fold_epoch(prepared, stats);
    const bool rolled_back = complete_execution<UseCycle>(
        prepared, pool.authority(), cycle, false, true, false);
    prepared.token = 0u;
    poison = !rolled_back || !folded;
    return folded ? retained : folded;
  }
  if constexpr (UseAccelerator) {
    if (!run.reduction) {
      const BufferReadView view = residency_output_view(*prepared.pipeline);
      const residency::EpochLease outputs = execution_output_lease(prepared);
      prepared.coherent_output = static_cast<bool>(view);
      for (std::size_t index = 0u;
           prepared.coherent_output && index < outputs.bindings.size();
           ++index) {
        const residency::CacheBinding binding = outputs.bindings[index];
        const std::uint32_t first =
            pool.first_output_frame +
            prepared.pipeline->residency_bank * run.frame_capacity;
        if (binding.frame < first ||
            binding.frame - first >= run.frame_capacity) {
          prepared.coherent_output = false;
          break;
        }
        const std::uint64_t offset =
            static_cast<std::uint64_t>(binding.frame - first) *
            run.output_page_bytes;
        if (offset > view.bytes ||
            run.output_page_bytes > view.bytes - offset) {
          prepared.coherent_output = false;
          break;
        }
        prepared.output_frames[index] = view.data + offset;
      }
    }
  }
  const bool execution_completed =
      complete_execution<UseCycle>(prepared, pool.authority(), cycle, true);
  if (!execution_completed) {
    const Status folded = fold_epoch(prepared, stats);
    const bool cleaned = complete_execution<UseCycle>(
        prepared, pool.authority(), cycle, false, true, true);
    prepared.token = 0u;
    poison = true;
    return !folded ? folded
                   : Status::fail(cleaned ? Reason::PipelineInvalid
                                          : Reason::PipelineBusy);
  }
  prepared.token = 0u;
  const Status reserved = reserve_drain();
  if (reserved) {
    return reserved;
  }
  const Status folded = fold_epoch(prepared, stats);
  return folded ? reserved : folded;
}

bool complete_cpu_epoch(PreparedEpoch &prepared,
                        residency::Authority &authority,
                        const std::uint64_t cycle, const bool success,
                        const bool invalidate_cycle,
                        const bool invalidate_direct) noexcept {
  return complete_execution<false>(prepared, authority, cycle, success,
                                   invalidate_cycle, invalidate_direct);
}

bool complete_accel_epoch(PreparedEpoch &prepared,
                          residency::Authority &authority,
                          const std::uint64_t cycle, const bool success,
                          const bool use_cycle, const bool invalidate_cycle,
                          const bool invalidate_direct) noexcept {
  return use_cycle
             ? complete_execution<true>(prepared, authority, cycle, success,
                                        invalidate_cycle, invalidate_direct)
             : complete_execution<false>(prepared, authority, cycle, success,
                                         invalidate_cycle, invalidate_direct);
}

Status finish_cpu_epoch(PreparedEpoch &prepared,
                        const VirtualRunProjection &run, Stats &stats,
                        const std::uint64_t cycle, bool &poison) noexcept {
  return finish_epoch<false, false>(prepared, run, stats, cycle, poison);
}

Status finish_accel_epoch(PreparedEpoch &prepared,
                          const VirtualRunProjection &run, Stats &stats,
                          const std::uint64_t cycle, const bool use_cycle,
                          bool &poison) noexcept {
  return use_cycle
             ? finish_epoch<true, true>(prepared, run, stats, cycle, poison)
             : finish_epoch<true, false>(prepared, run, stats, cycle, poison);
}

} // namespace rund::compute::detail::virtual_run_overlap
