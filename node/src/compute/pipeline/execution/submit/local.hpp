#pragma once

#include "../submit.hpp"

namespace rund::compute::detail {

void clear_pipeline_execution_control(
    residency::execution::Control &control) noexcept;
void complete_pipeline_execution_callback(
    void *, node::accel::detail::PreparedPipelineEvidence &&) noexcept;

} // namespace rund::compute::detail
