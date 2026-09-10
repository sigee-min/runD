#pragma once

#include "../run.hpp"

namespace rund::node::accel::detail::backend_run_detail {

[[nodiscard]] bool bind_primitive(const KernelExecutionStep &, const RunBinds &,
                                  BoundBindings &);
[[nodiscard]] bool bind_map(const rund::AccelContext &, BoundStep &);
[[nodiscard]] bool bind_control(const KernelExecutionStep &, const RunBinds &,
                                BoundControl &) noexcept;

} // namespace rund::node::accel::detail::backend_run_detail
