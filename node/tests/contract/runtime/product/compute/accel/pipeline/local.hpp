#pragma once

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

namespace runtime_compute_pipeline_accel_detail {

[[nodiscard]] int CheckPipelineControl(rund::Session &,
                                       rund::compute::Device &);
[[nodiscard]] int CheckFixedRecurrences(rund::compute::Device &);

} // namespace runtime_compute_pipeline_accel_detail
