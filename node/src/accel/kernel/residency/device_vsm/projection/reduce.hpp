#pragma once

#include "../projection.hpp"

#include "../../../prepared/model.hpp"

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/reduce/model.hpp>

namespace rund::node::accel::detail::device_vsm_reduce_projection {

struct Authority final {
  rund::kernel::ReducePlan semantic{};
  rund::kernel::ComputeApi api{rund::kernel::ComputeApi::Cpu};
};

[[nodiscard]] bool exact(const prepared::PipelineState &, Authority &,
                         const char *&reason) noexcept;
[[nodiscard]] bool exact_graph(const prepared::PipelineState &, Authority &,
                               const char *&reason) noexcept;
[[nodiscard]] bool same(const prepared::PipelineState &,
                        const Authority &) noexcept;

} // namespace rund::node::accel::detail::device_vsm_reduce_projection
