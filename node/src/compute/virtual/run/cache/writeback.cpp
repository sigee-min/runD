#include "internal.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] bool
valid_transaction_sink(const VirtualBacking &output,
                       const VirtualRunProjection &run,
                       const VirtualRunTransaction &transaction) noexcept {
  if (!transaction.started || !transaction.scan || !transaction.output ||
      transaction.output != &output || transaction.provider == nullptr ||
      !transaction.token ||
      dynamic_cast<VirtualBackingTransaction *>(transaction.output) !=
          transaction.provider ||
      !run.scan() || run.graph_execution() || run.reduction()) {
    return false;
  }
  const VirtualBackingTransactionSpec spec = transaction.token.spec();
  return spec.backing_id != 0u && spec.base_version != 0u &&
         spec.logical_bytes == output.size_bytes() &&
         spec.logical_bytes == run.active.output_bytes &&
         spec.backing_id == VirtualBackingAccess::id(output) &&
         spec.base_version == VirtualBackingAccess::version(output) &&
         spec.backing_page_bytes == run.output_page_bytes &&
         spec.write_page_bytes == run.output_payload_bytes &&
         spec.write_page_count == transaction.page_count &&
         spec.backing_page_count != 0u &&
         VirtualBackingAccess::recovery_bytes(output) >= spec.logical_bytes;
}

} // namespace

Status stage_residency_cache(
    VirtualBacking &output, const VirtualRunProjection &run,
    const std::span<const residency::CacheTransition> transitions,
    ResidencyStats &residency_stats, VirtualTransferInterval *const interval,
    const std::span<const std::byte *const> frames,
    VirtualRunTransaction *const transaction) noexcept {
  const bool local_transaction =
      transaction != nullptr && transaction->started && transaction->scan;
  if (local_transaction && !valid_transaction_sink(output, run, *transaction)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::size_t writeback_count = 0u;
  for (const residency::CacheTransition &transition : transitions) {
    writeback_count += static_cast<std::size_t>(
        transition.kind == residency::TransitionKind::Writeback);
  }
  if (writeback_count == 0u) {
    return Status::success();
  }
  if (writeback_count > PipelineLeafCapacity ||
      (!frames.empty() && frames.size() != writeback_count)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::array<VirtualWrite, PipelineLeafCapacity> ranges{};
  std::size_t count = 0u;
  std::uint64_t logical_bytes = 0u;
  VirtualBackingAccess::require_recovery(output, run.active.output_bytes);
  for (const residency::CacheTransition &transition : transitions) {
    if (transition.kind != residency::TransitionKind::Writeback) {
      continue;
    }
    if (transition.key.domain != residency::CacheDomain::Backing ||
        transition.dirty.empty() ||
        transition.key.page > std::numeric_limits<std::uint64_t>::max() /
                                  run.output_payload_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::uint64_t page_offset =
        transition.key.page * run.output_payload_bytes;
    if (transition.dirty.offset < page_offset ||
        transition.dirty.offset >= run.active.output_bytes ||
        transition.dirty.bytes >
            run.active.output_bytes - transition.dirty.offset ||
        transition.dirty.offset - page_offset > run.output_payload_bytes ||
        transition.dirty.bytes > run.output_payload_bytes -
                                     (transition.dirty.offset - page_offset)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = static_cast<std::size_t>(transition.dirty.bytes);
    const std::size_t frame_offset = static_cast<std::size_t>(
        run.output_prefix_bytes + transition.dirty.offset - page_offset);
    const std::byte *const frame =
        frames.empty() ? virtual_resident_output_frame(run, transition.frame)
                       : frames[count];
    if (frame == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    ranges[count++] = VirtualWrite{
        .offset = transition.dirty.offset,
        .bytes = std::span<const std::byte>{frame + frame_offset, bytes},
    };
    Accumulate(logical_bytes, bytes);
  }
  const std::uint64_t started = pipeline_clock();
  const Status written = [&]() noexcept {
    if (local_transaction) {
      return transaction->provider->stage(
          transaction->token,
          std::span<const VirtualWrite>{ranges.data(), count});
    }
    VirtualBackingTransaction *const legacy_provider =
        VirtualBackingAccess::transaction_provider(output);
    VirtualBackingTransactionToken *const legacy_token =
        VirtualBackingAccess::transaction_token(output);
    return legacy_provider != nullptr && legacy_token != nullptr
               ? legacy_provider->stage(
                     *legacy_token,
                     std::span<const VirtualWrite>{ranges.data(), count})
               : output.write_batch(
                     std::span<const VirtualWrite>{ranges.data(), count});
  }();
  const std::uint64_t completed = pipeline_clock();
  if (interval != nullptr) {
    *interval = VirtualTransferInterval{.started_ns = started,
                                        .completed_ns = completed};
  }
  Accumulate(residency_stats.backing_io_ns, completed - started);
  if (!written) {
    return written;
  }
  Accumulate(residency_stats.backing_write_bytes, logical_bytes);
  Accumulate(residency_stats.page_out_bytes, logical_bytes);
  Accumulate(residency_stats.page_out_count, count);
  return Status::success();
}

void publish_residency_cache(
    VirtualBacking &output, VirtualRunTransaction *const transaction) noexcept {
  if (transaction != nullptr && transaction->started && transaction->scan) {
    return;
  }
  if (VirtualBackingAccess::transaction_provider(output) != nullptr) {
    return;
  }
  VirtualBackingAccess::publish_write(output);
}

namespace {

[[nodiscard]] bool has_writeback_transition(
    const std::span<const residency::CacheTransition> transitions) noexcept {
  return std::any_of(transitions.begin(), transitions.end(),
                     [](const residency::CacheTransition &transition) {
                       return transition.kind ==
                              residency::TransitionKind::Writeback;
                     });
}

} // namespace

Status writeback_residency_cache(
    VirtualBacking &output, const VirtualRunProjection &run,
    const std::span<const residency::CacheTransition> transitions,
    ResidencyStats &residency_stats, VirtualTransferInterval *const interval,
    const std::span<const std::byte *const> frames,
    VirtualRunTransaction *const transaction) noexcept {
  const Status staged = stage_residency_cache(
      output, run, transitions, residency_stats, interval, frames, transaction);
  if (staged && has_writeback_transition(transitions)) {
    publish_residency_cache(output, transaction);
  }
  return staged;
}

} // namespace rund::compute::detail
