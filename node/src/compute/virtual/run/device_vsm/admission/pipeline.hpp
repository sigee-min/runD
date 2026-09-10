#pragma once

#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail::admission_detail {

[[nodiscard]] bool
pipeline_authority(const std::shared_ptr<PipelineState> &selected_primary,
                   const std::shared_ptr<PipelineState> &second,
                   const VirtualRunProjection &run, bool multi_pointwise,
                   bool multi_scan) noexcept;

} // namespace
  // rund::compute::detail::device_vsm_product_detail::admission_detail
