#pragma once

#include <rund/compute/pipeline/coordinate.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t PreparedPipelineUnknownCoordinate =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr char PreparedPipelineTemplateStepCapacityReasonKey[] =
    "compute_pipeline_template_step_capacity";

// Cold preparation is intentionally split into stable, backend-neutral
// stages.  A backend may refine route coordinates only while it owns the
// corresponding canonical template, physical occurrence, or graph node.
enum class PreparedPipelineFailureStage : std::uint8_t {
  Unknown,
  CommonValidation,
  CommonExpansion,
  CommonAccounting,
  BackendAdmission,
  BackendDescription,
  BackendAllocation,
  BackendCapture,
  BackendFinalization,
};

struct PreparedPipelineFailure final {
  PreparedPipelineFailureStage stage{PreparedPipelineFailureStage::Unknown};
  std::uint32_t template_index{PreparedPipelineUnknownCoordinate};
  std::uint32_t occurrence_index{PreparedPipelineUnknownCoordinate};
  std::uint32_t node{PreparedPipelineUnknownCoordinate};
  std::uint32_t outer_iteration{PreparedPipelineUnknownCoordinate};
  std::uint32_t inner_iteration{PreparedPipelineUnknownCoordinate};
  rund::compute::PipelineNestedPhase nested_phase{
      rund::compute::PipelineNestedPhase::None};
  const char *native_reason_key = "accel_kernel_pipeline_invalid";
};

} // namespace rund::node::accel::detail
