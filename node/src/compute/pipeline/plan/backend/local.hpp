#pragma once

#include "../local.hpp"

#include "../../../../accel/kernel/prepared/pipeline.hpp"

#include <string_view>

namespace rund::compute::detail {

[[nodiscard]] Reason
project_pipeline_preparation_reason(std::string_view reason) noexcept;

[[nodiscard]] Result<node::accel::detail::PreparedKernelPipeline>
prepare_backend_stream(PipelineState &state, bool alternate,
                       Location &location);

} // namespace rund::compute::detail
