#pragma once

#include "state.hpp"

namespace rund::compute::detail {

void record_write(JobState &state, WriteStats stats) noexcept;

} // namespace rund::compute::detail
