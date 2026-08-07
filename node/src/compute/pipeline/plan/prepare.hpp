#pragma once

#include "../output.hpp"
#include "contract.hpp"

#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail {

struct PipelineBuildState;
struct PipelineMemoryPlan;
struct PipelineState;
struct PipelineScheduleSuccess final {};

struct PipelinePrepare final {
  std::shared_ptr<PipelineState> state;
  PipelineHash hash{};
  std::size_t output_count{};
  std::uint64_t status_count{};
  Location failure{};
};

[[nodiscard]] Status
admit_pipeline(const std::shared_ptr<PipelineBuildState> &build,
               PipelinePrepare &prepare);
[[nodiscard]] Result<PipelineScheduleSuccess>
plan_pipeline_schedule(const PipelineBuildState &build,
                       PipelineMemoryPlan &plan);
[[nodiscard]] Status
schedule_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                  PipelinePrepare &prepare);
[[nodiscard]] Status
bind_pipeline(const std::shared_ptr<PipelineBuildState> &build,
              PipelinePrepare &prepare);

} // namespace rund::compute::detail
