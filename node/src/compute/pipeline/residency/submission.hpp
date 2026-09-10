#pragma once

#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::compute::detail {

struct PipelineState;

using ResidencyPipelineCompletion = void (*)(void *, Status) noexcept;

enum class ResidencySubmissionPhase : std::uint8_t {
  Empty,
  Submitting,
  Submitted,
  Completed,
};

// Fixed transaction storage for one asynchronous accelerator VSM epoch. The
// selected local order is borrowed by the prepared backend until its native
// completion callback; keeping it beside the retained Pipeline owner prevents
// a caller stack or Authority lease view from becoming a second lifetime
// authority. Executor::wait() is the only reset boundary.
struct ResidencyPipelineSubmission final {
  std::shared_ptr<PipelineState> pipeline{};
  std::array<std::uint32_t, PipelineLeafCapacity> locals{};
  std::size_t issued_steps{};
  bool writes_possible{};
  // Aggregate ordinary Scan keeps physical control identities private until
  // the run-level cursor commits them.  The canonical publication identity is
  // carried separately so a deferred terminal still validates its base.
  bool defer_generation{};
  std::uint64_t control_generation{
      std::numeric_limits<std::uint64_t>::max()};
  std::uint8_t control_parity{};
  std::uint64_t publication_generation{
      std::numeric_limits<std::uint64_t>::max()};
  std::uint8_t publication_parity{};
  ResidencyPipelineCompletion completion{};
  void *user{};
  std::atomic<ResidencySubmissionPhase> phase{ResidencySubmissionPhase::Empty};

  [[nodiscard]] bool active() const noexcept {
    return phase.load(std::memory_order_acquire) !=
           ResidencySubmissionPhase::Empty;
  }
  void reset() noexcept {
    pipeline.reset();
    locals.fill(0u);
    issued_steps = 0u;
    writes_possible = false;
    defer_generation = false;
    control_generation = std::numeric_limits<std::uint64_t>::max();
    control_parity = 0u;
    publication_generation = std::numeric_limits<std::uint64_t>::max();
    publication_parity = 0u;
    completion = nullptr;
    user = nullptr;
    phase.store(ResidencySubmissionPhase::Empty, std::memory_order_release);
  }
};

} // namespace rund::compute::detail
