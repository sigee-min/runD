#pragma once

#include "../../execution.hpp"
#include "../../backing.hpp"

#include "../../../../device/residency/execution/run.hpp"
#include "../../../../pipeline/execution/window.hpp"
#include "../../../../pipeline/state.hpp"
#include "../../../stats.hpp"
#include "../../../../../hash/fnv.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {

struct WindowWait final {
  std::mutex coordinator{};
  std::mutex gate{};
  std::condition_variable ready{};
  residency::execution::Window *window{};
  PipelineResidencyWindowControl *native{};
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
  std::array<bool, residency::execution::WindowCapacity> input_done{};
  std::array<bool, residency::execution::WindowCapacity> output_done{};
  VirtualInputReuseSeed input_reuse{};
  ::rund::node::hash_detail::Fnv hash{};
  residency::execution::WindowEvidence final{};
  Status failure{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t output_hash{};
  bool poison{};
  bool completed{};
};

[[nodiscard]] Status ServiceWindowInput(
    WindowWait &wait, std::uint64_t epoch) noexcept;
[[nodiscard]] Status ServiceWindowInput(
    WindowWait &wait, std::uint64_t epoch,
    const residency::ExecutionTicket &ticket) noexcept;
void CompleteWindowNative(
    void *raw, PipelineWindowRelease &&pipeline) noexcept;
void CompleteWindowFinal(
    void *raw, residency::execution::WindowEvidence &&evidence) noexcept;

} // namespace rund::compute::detail
