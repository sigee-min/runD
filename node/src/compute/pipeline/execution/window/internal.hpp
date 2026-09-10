#pragma once

#include "../window.hpp"

namespace rund::compute::detail::pipeline_window_detail {

void complete_release(
    void *, node::accel::detail::PreparedResidencyWindowRelease &&) noexcept;
void complete_final(
    void *, node::accel::detail::PreparedResidencyWindowFinal &&) noexcept;

} // namespace rund::compute::detail::pipeline_window_detail
