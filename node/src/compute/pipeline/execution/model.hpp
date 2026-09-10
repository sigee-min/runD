#pragma once

#include "../../device/residency/execution/owner.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

struct PipelineState;

enum class PipelineExecutionPhase : std::uint8_t {
  Empty,
  Submitting,
  Submitted,
  Completed,
};

// Fixed warm owner for one Q=1 run-level native Dispatch. It retains the
// exact PipelineState and selected locals until the raw prepared callback has
// passed through Pipeline terminal publication. Authority credentials remain
// caller-owned in execution::Control and are snapshotted before submission.
struct PipelineExecutionSubmission final {
  std::shared_ptr<PipelineState> pipeline{};
  residency::execution::Control *control{};
  std::array<std::uint32_t, residency::execution::UseCapacity> locals{};
  std::size_t issued_steps{};
  bool writes_possible{};
  std::atomic<PipelineExecutionPhase> phase{PipelineExecutionPhase::Empty};

  [[nodiscard]] bool active() const noexcept {
    return phase.load(std::memory_order_acquire) !=
           PipelineExecutionPhase::Empty;
  }

  // The caller may reset only after the completion callback has entered and
  // the submission has no remaining access to this owner. The common
  // execution join observes that terminal before reusing this fixed slot.
  void reset() noexcept {
    pipeline.reset();
    control = nullptr;
    locals.fill(0u);
    issued_steps = 0u;
    writes_possible = false;
    phase.store(PipelineExecutionPhase::Empty, std::memory_order_release);
  }
};

} // namespace rund::compute::detail
