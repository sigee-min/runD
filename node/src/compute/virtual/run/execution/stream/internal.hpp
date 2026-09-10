#pragma once

#include "../../backing.hpp"
#include "../../execution.hpp"

#include "../../../../../accel/kernel/prepared/interface/api.hpp"
#include "../../../../device/residency/execution/stream.hpp"
#include "../../../../pipeline/execution/schedule.hpp"
#include "../../../../pipeline/execution/window.hpp"
#include "../../../../../hash/fnv.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace rund::compute::detail::virtual_stream_detail {

// One source-private controller for the whole recurrent execution. All
// controller facts remain behind the coordinator mutex; `gate` only carries
// the final publication to the caller. The split stream owners below mutate
// this same object and never retain a second journal, claim, or state mirror.
struct StreamWait final {
  std::mutex coordinator{};
  std::mutex gate{};
  std::condition_variable ready{};
  residency::execution::Stream *stream{};
  const residency::execution::Plan *plan{};
  PipelineResidencyWindowControl *native{};
  PipelineResidencyScheduleControl *schedule{};
  node::accel::detail::PreparedResidencyStreamControl *claim{};
  residency::Pool *pool{};
  VirtualPipelineState *state{};
  VirtualBacking *input{};
  VirtualBacking *output{};
  const VirtualRunProjection *run{};
  Stats *stats{};
  std::array<std::shared_ptr<PipelineState>, residency::execution::BankCapacity>
      pipelines{};
  std::array<std::array<const std::byte *, PipelineLeafCapacity>,
             residency::execution::WindowCapacity>
      output_storage{};
  std::array<std::span<const std::byte *const>,
             residency::execution::WindowCapacity>
      output_frames{};
  VirtualInputReuseSeed input_reuse{};
  ::rund::node::hash_detail::Fnv hash{};
  residency::execution::StreamEvidence final{};
  // `disposition`, `quarantine`, and `terminal` are controller facts guarded
  // by `coordinator`. The similarly named public outcome below is copied once
  // under `gate` before `completed` and the condition-variable notification.
  Status disposition{Status::success()};
  bool quarantine{};
  bool terminal{};
  Status failure{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t chunk_first{};
  std::size_t chunk_count{};
  std::thread::id caller{};
  bool callback_on_caller{};
  bool poison{};
  bool pending{};
  bool scheduled{};
  bool pumping{};
  bool completed{};
};

[[nodiscard]] Status service_stream_input(
    StreamWait &, std::uint64_t,
    const residency::ExecutionTicket *issued = nullptr) noexcept;
[[nodiscard]] Status service_stream_output(StreamWait &, std::uint64_t) noexcept;

[[nodiscard]] Status signal_stream_native(StreamWait &, std::uint64_t,
                                          Status) noexcept;
[[nodiscard]] Status abort_stream_native(StreamWait &, Status) noexcept;

void complete_stream_native(void *, PipelineWindowRelease &&) noexcept;
void complete_stream_schedule_final(
    void *, residency::execution::ScheduleEvidence &&) noexcept;
void complete_stream_final(void *, residency::execution::WindowEvidence &&)
    noexcept;

void pump_stream(StreamWait &) noexcept;
void pump_stream_task(void *) noexcept;

} // namespace rund::compute::detail::virtual_stream_detail
