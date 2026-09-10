#pragma once

namespace rund::compute::detail::device_vsm_product_detail {

struct DeviceVsmProductRun;

[[nodiscard]] bool reuse_output_hash(DeviceVsmProductRun &) noexcept;
void observe_output_hash(DeviceVsmProductRun &) noexcept;
void commit_output_hash(DeviceVsmProductRun &) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
