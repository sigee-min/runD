#pragma once

#include <cstdint>

namespace rund::kernel {

struct ComputePlan;

} // namespace rund::kernel

namespace rund::node::accel::detail {

struct KernelExecutionStep;

} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] bool map_source_upper(const KernelExecutionStep &step,
                                    const rund::kernel::ComputePlan &plan,
                                    std::uint64_t &retained,
                                    std::uint64_t &transient) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
