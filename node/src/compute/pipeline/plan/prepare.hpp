#pragma once

#include "contract.hpp"
#include "output.hpp"

#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail {

struct PipelineBuildState;
struct PipelineState;

struct PipelinePrepare final {
  std::shared_ptr<PipelineState> state;
  std::vector<PipelinePlanStep> steps;
  std::vector<PipelineResourceAdmission> admissions;
  std::array<PhysicalOutputProjection, PipelineIterationCapacity> outputs{};
  PipelineHash hash{};
  std::size_t output_count{};
  std::size_t binding_count{};
  std::uint64_t status_count{};
  Location failure{};
};

[[nodiscard]] Status
admit_pipeline(const std::shared_ptr<PipelineBuildState> &build,
               PipelinePrepare &prepare);
[[nodiscard]] Status
schedule_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                  PipelinePrepare &prepare);
[[nodiscard]] Status
bind_pipeline(const std::shared_ptr<PipelineBuildState> &build,
              PipelinePrepare &prepare);

} // namespace rund::compute::detail
