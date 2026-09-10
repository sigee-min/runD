#pragma once

#include "../state.hpp"

#include "../../../accel/kernel/prepared/interface/evidence.hpp"

#include <cstddef>
#include <optional>

namespace rund::compute::detail {

enum class PipelineNativeCompletion : std::uint8_t {
  NotSubmitted,
  Known,
  UnknownMayWrite,
};

struct PipelineOutcome final {
  Status status{Status::success()};
  std::size_t verified{};
  std::optional<std::size_t> failed_step{};
  bool writes_possible{};
  PipelineNativeCompletion native_completion{
      PipelineNativeCompletion::NotSubmitted};
  bool publication_suppressed{};
  // Aggregate ordinary Scan defers canonical generation until its run-level
  // two-bank cursor commits every successful physical terminal.
  bool defer_generation{};

  [[nodiscard]] bool submitted() const noexcept {
    return native_completion != PipelineNativeCompletion::NotSubmitted;
  }
  [[nodiscard]] bool unknown_terminal() const noexcept {
    return native_completion == PipelineNativeCompletion::UnknownMayWrite;
  }
};

void reset_pipeline_stats(PipelineState &state) noexcept;

[[nodiscard]] Status pipeline_window_status(const PipelineWindows &windows,
                                            const PipelineStep &step,
                                            Status status,
                                            ControlStats &stats) noexcept;

[[nodiscard]] PipelineOutcome finish_accel_pipeline(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept;

} // namespace rund::compute::detail
