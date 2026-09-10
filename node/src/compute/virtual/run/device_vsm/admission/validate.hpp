#pragma once

#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail::admission_detail {

[[nodiscard]] ::rund::AccelCheck
check_bindings(const VirtualRunProjection &, std::uint64_t page_count,
               std::size_t element_bytes, bool graph_pointwise,
               bool graph_map_reduce, bool scan, bool reduce,
               const std::shared_ptr<PipelineState> &selected_primary,
               const std::shared_ptr<PipelineState> &second,
               bool authority) noexcept;

} // namespace
  // rund::compute::detail::device_vsm_product_detail::admission_detail
