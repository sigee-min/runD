#pragma once

#include "../local.hpp"

#include <cstdint>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSurfaceRecord(
    rund::compute::Device &, const rund::compute::Buffer<std::int32_t> &);
[[nodiscard]] int CheckSurfaceBounded(
    rund::compute::Device &, const rund::compute::Buffer<std::int32_t> &);
[[nodiscard]] int CheckSurfaceAlias(
    rund::compute::Device &, const rund::compute::Buffer<std::int32_t> &);
[[nodiscard]] int CheckSurfaceSharedCount(rund::compute::Device &);

} // namespace rund_node_test_pipeline
