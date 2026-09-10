#pragma once

#include "../schedule.hpp"

namespace rund::compute::detail {

void complete_schedule_release(
    void *, node::accel::detail::PreparedResidencyScheduleRelease &&) noexcept;
void complete_schedule_final(
    void *, node::accel::detail::PreparedResidencyScheduleFinal &&) noexcept;

} // namespace rund::compute::detail
