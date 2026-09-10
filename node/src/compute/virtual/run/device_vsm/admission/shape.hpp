#pragma once

#include "../internal.hpp"

#include <cstddef>

namespace rund::compute::detail::device_vsm_product_detail::admission_detail {

struct Shape final {
  bool pointwise{};
  bool multi_pointwise{};
  bool multi_scan{};
  bool window{};
  std::size_t stages{};
  bool graph_map_reduce{};
  bool graph_pointwise{};
  bool scan{};
  bool reduce{};
};

[[nodiscard]] Shape classify(const VirtualPipelineState &,
                             const VirtualRunProjection &,
                             std::size_t element_bytes) noexcept;

} // namespace
  // rund::compute::detail::device_vsm_product_detail::admission_detail
