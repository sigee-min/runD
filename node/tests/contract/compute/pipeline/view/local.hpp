#pragma once

#include "../local.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/math.hpp>

#include "src/compute/pipeline/state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund_node_test_pipeline::view {

[[nodiscard]] std::size_t
CpuViewTransferCount(const rund::compute::detail::JobState &) noexcept;
[[nodiscard]] const rund::compute::detail::BufferState *
FirstCpuViewBuffer(const rund::compute::detail::JobState &) noexcept;

[[nodiscard]] int CheckBasic(rund::compute::Device &, rund::compute::Backend);
[[nodiscard]] int CheckReduce(rund::compute::Device &, rund::compute::Backend);
[[nodiscard]] int CheckSort(rund::compute::Device &, rund::compute::Backend);
[[nodiscard]] int CheckPoolReset(rund::compute::Device &);
[[nodiscard]] int CheckPrimitives(rund::compute::Device &,
                                  rund::compute::Backend);
[[nodiscard]] bool CheckStridedReset(rund::compute::Device &,
                                     rund::compute::Backend);
[[nodiscard]] bool CheckScatterOffset(rund::compute::Device &);

} // namespace rund_node_test_pipeline::view
