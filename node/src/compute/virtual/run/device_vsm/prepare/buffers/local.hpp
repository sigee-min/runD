#pragma once

#include "../../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] bool
same_resident(const std::shared_ptr<VirtualBufferState> &buffer,
              const std::shared_ptr<BufferState> &resident) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
