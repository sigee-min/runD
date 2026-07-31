#pragma once

#include "../state.hpp"

#include "../../../accel/kernel/prepared.hpp"

#include <cstddef>

namespace rund::compute::detail {

struct PipelineOutcome final {
  Status status{Status::success()};
  std::size_t verified{};
  std::size_t failed_step{};
  bool failure_step_known{};
  bool writes_possible{};
  bool submitted{};
  bool publication_suppressed{};
};

void reset_pipeline_stats(PipelineState &state) noexcept;

[[nodiscard]] Status pipeline_window_status(PipelineState &state,
                                            const PipelineStep &step,
                                            Status status,
                                            Stats &stats) noexcept;

[[nodiscard]] PipelineOutcome finish_accel_pipeline(
    PipelineState &state,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept;

} // namespace rund::compute::detail
