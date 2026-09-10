#include "internal.hpp"

#include "../../../stats.hpp"
#include "../../cache.hpp"
#include "../fill.hpp"
#include "../io.hpp"

#include "../../../../device/residency/execution/run.hpp"
#include "../../../../device/residency/execution/sliding.hpp"
#include "../../../../pipeline/local.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail::virtual_stream_detail {
using ::rund::detail::counter::Accumulate;

[[nodiscard]] Status
service_stream_input(StreamWait &wait, const std::uint64_t epoch,
                     const residency::ExecutionTicket *const issued) noexcept {
  using namespace residency::execution;
  if (wait.stream == nullptr || wait.input == nullptr || wait.run == nullptr ||
      wait.stats == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::ExecutionTicket ticket{};
  if (issued == nullptr) {
    if (!wait.stream->issue(epoch, Phase::Input, ticket)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    ticket = *issued;
  }
  const std::size_t bank = epoch % BankCapacity;
  std::uint64_t backing_pages = 0u;
  std::uint64_t device_pages = 0u;
  Status status = execution_io::fill_input(
      *wait.input, *wait.run, ticket, wait.pipelines[bank].get(),
      wait.stats->pipeline.residency, backing_pages, &wait.input_reuse);
  if (status) {
    status = execution_io::supply_input(*wait.pipelines[bank], *wait.run,
                                        ticket, *wait.stats, device_pages);
  }
  const residency::ExecutionTerminal terminal =
      status ? residency::ExecutionTerminal::Success
             : residency::ExecutionTerminal::Failure;
  if (!wait.stream->terminal(ticket, terminal, status)) {
    status = Status::fail(Reason::PipelineInvalid);
  }
  if (!status) {
    wait.failed_page = execution_io::failed_page(ticket);
    return status;
  }
  const std::size_t count = ticket.bindings.size() / 2u;
  const residency::FrameRegion device_region = wait.run->input_regions[bank];
  std::uint64_t evictions = 0u;
  for (const residency::CacheTransition &transition : ticket.transitions) {
    evictions += static_cast<std::uint64_t>(
        transition.kind == residency::TransitionKind::Unmap &&
        transition.frame >= device_region.first &&
        transition.frame - device_region.first < device_region.count);
  }
  Accumulate(wait.stats->pipeline.residency.page_in_count, device_pages);
  Accumulate(wait.stats->pipeline.residency.cache_hit_count,
             count - device_pages);
  Accumulate(wait.stats->pipeline.residency.eviction_count, evictions);
  Accumulate(wait.stats->pipeline.residency.page_in_bytes,
             device_pages * wait.run->input_page_bytes);
  Accumulate(wait.stats->pipeline.residency.late_page_count, backing_pages);
  return Status::success();
}

[[nodiscard]] Status service_stream_output(StreamWait &wait,
                                           const std::uint64_t epoch) noexcept {
  using namespace residency::execution;
  residency::ExecutionTicket ticket{};
  if (wait.stream == nullptr || wait.output == nullptr || wait.run == nullptr ||
      wait.stats == nullptr ||
      !wait.stream->issue(epoch, Phase::Output, ticket)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t slot = epoch % WindowCapacity;
  const std::size_t bank = epoch % BankCapacity;
  Status status = retain_residency_output(*wait.pipelines[bank], *wait.run,
                                          ticket, *wait.stats);
  if (status) {
    status = execution_io::collect_output(*wait.pipelines[bank], *wait.run,
                                          ticket, wait.output_frames[slot],
                                          wait.output_storage[slot]);
  }
  if (status) {
    status = stage_residency_cache(*wait.output, *wait.run, ticket.transitions,
                                   wait.stats->pipeline.residency, nullptr,
                                   wait.output_frames[slot]);
  }
  const residency::ExecutionTerminal terminal =
      status ? residency::ExecutionTerminal::Success
             : residency::ExecutionTerminal::Failure;
  if (!wait.stream->terminal(ticket, terminal, status)) {
    status = Status::fail(Reason::PipelineInvalid);
  }
  if (!status) {
    wait.failed_page = execution_io::failed_page(ticket);
    return status;
  }
  for (std::size_t local = 0u; local < wait.output_frames[slot].size();
       ++local) {
    const std::uint64_t page = epoch * wait.run->frame_capacity + local;
    const std::uint64_t logical = page * wait.run->output_payload_bytes;
    const std::size_t bytes = static_cast<std::size_t>(
        std::min(wait.run->output_payload_bytes,
                 wait.run->active.output_bytes - logical));
    wait.hash.Bytes(
        reinterpret_cast<const std::uint8_t *>(wait.output_frames[slot][local] +
                                               wait.run->output_prefix_bytes),
        bytes);
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_stream_detail
