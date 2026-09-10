#pragma once

#include "../local.hpp"

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckBoundedRepeat(rund::compute::Device &);
[[nodiscard]] int CheckResetRepeat(rund::compute::Device &, Backend);

} // namespace rund_node_test_pipeline
