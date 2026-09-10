#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../graph/reduce/timeline.hpp"
#include "../../stats.hpp"
#include "../backing.hpp"
#include "../overlap.hpp"
#include "../reduce.hpp"

#include <rund/compute/pipeline/runtime.hpp>

#include <rund/counter.hpp>

#include <array>

namespace rund::compute::detail::virtual_run_overlap {
namespace {

using ::rund::detail::counter::Accumulate;

template <bool UseAccelerator, bool UseCycle>
[[nodiscard]] VirtualEpochResult
execute_overlap_impl(VirtualPipelineState &state, VirtualBacking &input,
                     VirtualBacking &output, const VirtualRunProjection &run,
                     Stats &stats, ::rund::node::hash_detail::Fnv &output_hash,
                     VirtualReduction *const reduction) noexcept {
  static_assert(UseAccelerator || !UseCycle);
  residency::Pool &pool = *state.pipeline->residency_pool;
  std::array<bool, 2u> prefetch_pending{};
  std::array<PreparedEpoch, 2u> tickets{};
  const std::uint64_t epochs = run.active.stream.epoch_count();
  bool poison = false;
  VirtualInputReuseSeed input_reuse{};
  [[maybe_unused]] std::uint64_t cycle_token = 0u;
  const auto cancel_all = [&]() noexcept {
    bool clean = true;
    if constexpr (UseCycle) {
      if (cycle_token != 0u) {
        clean = pool.authority().cycles().close_cycle(cycle_token) && clean;
      }
    }
    for (PreparedEpoch &ticket : tickets) {
      if (ticket.submitted) {
        const residency::ExecutionReceipt receipt =
            ticket.pipeline == nullptr
                ? residency::ExecutionReceipt{.status = Status::fail(
                                                  Reason::PipelineInvalid)}
                : pool.wait_execution(ticket.pipeline->residency_bank);
        ticket.submitted = false;
        const Status folded = fold_epoch(ticket, stats);
        clean = static_cast<bool>(receipt.status) &&
                static_cast<bool>(folded) &&
                !poisoned_pipeline(ticket.pipeline) && clean;
      }
    }
    for (PreparedEpoch &ticket : tickets) {
      if (ticket.drain_token != 0u) {
        clean = pool.authority().discard(ticket.drain_token) && clean;
        ticket.drain_token = 0u;
      }
      if (ticket.output.token != 0u) {
        clean = ticket.pipeline != nullptr &&
                cancel_residency_output(*ticket.pipeline, ticket.output) &&
                clean;
      }
      if (ticket.token != 0u) {
        const bool completed =
            UseAccelerator
                ? complete_accel_epoch(ticket, pool.authority(), cycle_token,
                                       false, UseCycle, true, true)
                : complete_cpu_epoch(ticket, pool.authority(), cycle_token,
                                     false, true, true);
        clean = completed && clean;
        ticket.token = 0u;
      }
    }
    // Cancellation waits for worker quiescence first. The target lease is
    // invalidated above; an Unknown/cancelled cohort conservatively discards
    // the source row after its exact alias credential is checked.
    clean = (UseAccelerator
                 ? wait_prefetch_accel(pool, prefetch_pending, false, false)
                 : wait_prefetch_cpu(pool, prefetch_pending, false, false)) &&
            clean;
    return clean;
  };
  TimelineInterval initial{};
  TimelineInterval initial_upload{};
  Status status =
      UseAccelerator
          ? prepare_accel_epoch(state, input, run, 0u, prefetch_pending, stats,
                                tickets[0], initial, initial_upload, true,
                                poison)
          : prepare_cpu_epoch(state, input, run, 0u, prefetch_pending, stats,
                              tickets[0], initial, initial_upload, true, poison,
                              &input_reuse);
  if (!status) {
    poison = !cancel_all() || poison;
    return VirtualEpochResult{.status = status,
                              .failed_page = tickets[0].projection.failed_page,
                              .poison_pipeline = poison};
  }
  if constexpr (UseCycle) {
    residency::cycle::Flight first{};
    if (!cycle_flight(tickets[0], run, first) ||
        !pool.authority().cycles().bind_cycle(first, cycle_token)) {
      poison = !cancel_all() || poison;
      return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                                .failed_page =
                                    tickets[0].projection.failed_page,
                                .poison_pipeline = poison};
    }
  }
  Accumulate(stats.pipeline.residency.stall_ns,
             graph_reduce::duration(initial));
  status = submit_epoch(tickets[0]);
  if (!status) {
    poison = !cancel_all() || poison;
    return VirtualEpochResult{.status = status,
                              .failed_page = tickets[0].projection.failed_page,
                              .poison_pipeline = poison};
  }
  if (epochs > 1u) {
    TimelineInterval next_ready{};
    TimelineInterval next_upload{};
    status = UseAccelerator
                 ? prepare_accel_epoch(state, input, run, 1u, prefetch_pending,
                                       stats, tickets[1], next_ready,
                                       next_upload, false, poison)
                 : prepare_cpu_epoch(state, input, run, 1u, prefetch_pending,
                                     stats, tickets[1], next_ready, next_upload,
                                     false, poison, &input_reuse);
    if (!status ||
        !graph_reduce::add_concurrent(tickets[0].timeline, next_ready) ||
        !graph_reduce::add_transfer(
            tickets[0].timeline, next_upload,
            graph_reduce::Timeline::Direction::HostToDevice)) {
      poison = !cancel_all() || poison;
      return VirtualEpochResult{
          .status = status ? Status::fail(Reason::PipelineInvalid) : status,
          .failed_page = tickets[1].projection.failed_page,
          .poison_pipeline = poison};
    }
    if constexpr (UseCycle) {
      residency::cycle::Flight next{};
      if (!cycle_flight(tickets[1], run, next) ||
          !pool.authority().cycles().advance_cycle(cycle_token, next)) {
        poison = !cancel_all() || poison;
        return VirtualEpochResult{
            .status = Status::fail(Reason::PipelineInvalid),
            .failed_page = tickets[1].projection.failed_page,
            .poison_pipeline = poison};
      }
      status = submit_epoch(tickets[1]);
      if (!status) {
        poison = !cancel_all() || poison;
        return VirtualEpochResult{.status = status,
                                  .failed_page =
                                      tickets[1].projection.failed_page,
                                  .poison_pipeline = poison};
      }
    }
  }

  for (std::uint64_t epoch = 0u; epoch < epochs; ++epoch) {
    PreparedEpoch &current = tickets[epoch % 2u];
    status = UseAccelerator
                 ? finish_accel_epoch(current, run, stats, cycle_token,
                                      UseCycle, poison)
                 : finish_cpu_epoch(current, run, stats, cycle_token, poison);
    if (!status) {
      poison = !cancel_all() || poison;
      return VirtualEpochResult{.status = status,
                                .failed_page = current.projection.failed_page,
                                .poison_pipeline = poison};
    }
    const bool has_next = epoch + 1u < epochs;
    PreparedEpoch *next = has_next ? &tickets[(epoch + 1u) % 2u] : nullptr;
    if (next != nullptr && !next->submitted) {
      status = submit_epoch(*next);
      if (!status) {
        poison = !cancel_all() || poison;
        return VirtualEpochResult{.status = status,
                                  .failed_page = next->projection.failed_page,
                                  .poison_pipeline = poison};
      }
    }
    PageOutTimeline page_out{};
    status = UseAccelerator
                 ? flush_accel_epoch(current, output, run, stats, output_hash,
                                     reduction, page_out, poison)
                 : flush_cpu_epoch(current, output, run, stats, output_hash,
                                   reduction, page_out, poison);
    if (!status) {
      poison = !cancel_all() || poison;
      return VirtualEpochResult{.status = status,
                                .failed_page = current.projection.failed_page,
                                .poison_pipeline = poison};
    }
    if (next == nullptr) {
      Accumulate(stats.pipeline.residency.stall_ns,
                 graph_reduce::duration(page_out.download));
      Accumulate(stats.pipeline.residency.stall_ns,
                 graph_reduce::duration(page_out.backing));
      continue;
    }
    const bool page_out_recorded =
        graph_reduce::add_concurrent(next->timeline, page_out.download) &&
        graph_reduce::add_concurrent(next->timeline, page_out.backing) &&
        graph_reduce::add_transfer(
            next->timeline, page_out.download,
            graph_reduce::Timeline::Direction::DeviceToHost);
    if (!page_out_recorded) {
      poison = !cancel_all() || poison;
      return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                                .failed_page = next->projection.failed_page,
                                .poison_pipeline = poison};
    }
    if (epoch + 2u < epochs) {
      TimelineInterval future_ready{};
      TimelineInterval future_upload{};
      status = UseAccelerator
                   ? prepare_accel_epoch(
                         state, input, run, epoch + 2u, prefetch_pending, stats,
                         current, future_ready, future_upload, false, poison)
                   : prepare_cpu_epoch(state, input, run, epoch + 2u,
                                       prefetch_pending, stats, current,
                                       future_ready, future_upload, false,
                                       poison, &input_reuse);
      const bool future_recorded =
          status &&
          graph_reduce::add_concurrent(next->timeline, future_ready) &&
          graph_reduce::add_transfer(
              next->timeline, future_upload,
              graph_reduce::Timeline::Direction::HostToDevice);
      if (!future_recorded) {
        poison = !cancel_all() || poison;
        return VirtualEpochResult{
            .status = status ? Status::fail(Reason::PipelineInvalid) : status,
            .failed_page = current.projection.failed_page,
            .poison_pipeline = poison};
      }
      if constexpr (UseCycle) {
        residency::cycle::Flight future{};
        if (!cycle_flight(current, run, future) ||
            !pool.authority().cycles().advance_cycle(cycle_token, future)) {
          poison = !cancel_all() || poison;
          return VirtualEpochResult{
              .status = Status::fail(Reason::PipelineInvalid),
              .failed_page = current.projection.failed_page,
              .poison_pipeline = poison};
        }
        status = submit_epoch(current);
        if (!status) {
          poison = !cancel_all() || poison;
          return VirtualEpochResult{.status = status,
                                    .failed_page =
                                        current.projection.failed_page,
                                    .poison_pipeline = poison};
        }
      }
    }
  }
  if constexpr (UseCycle) {
    if (!pool.authority().cycles().close_cycle(cycle_token)) {
      return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                                .poison_pipeline = true};
    }
  }
  if (!(UseAccelerator ? wait_prefetch_accel(pool, prefetch_pending, true)
                       : wait_prefetch_cpu(pool, prefetch_pending, true))) {
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .poison_pipeline = true};
  }
  return {};
}

} // namespace
} // namespace rund::compute::detail::virtual_run_overlap

namespace rund::compute::detail {

VirtualEpochResult
execute_virtual_cpu_overlap(VirtualPipelineState &state, VirtualBacking &input,
                            VirtualBacking &output,
                            const VirtualRunProjection &run, Stats &stats,
                            ::rund::node::hash_detail::Fnv &output_hash,
                            VirtualReduction *const reduction) noexcept {
  return virtual_run_overlap::execute_overlap_impl<false, false>(
      state, input, output, run, stats, output_hash, reduction);
}

VirtualEpochResult
execute_virtual_accel_overlap(VirtualPipelineState &state,
                              VirtualBacking &input, VirtualBacking &output,
                              const VirtualRunProjection &run, Stats &stats,
                              ::rund::node::hash_detail::Fnv &output_hash,
                              VirtualReduction *const reduction) noexcept {
  return run.reduction()
             ? virtual_run_overlap::execute_overlap_impl<true, false>(
                   state, input, output, run, stats, output_hash, reduction)
             : virtual_run_overlap::execute_overlap_impl<true, true>(
                   state, input, output, run, stats, output_hash, reduction);
}

} // namespace rund::compute::detail
