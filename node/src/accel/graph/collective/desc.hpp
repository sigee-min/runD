#pragma once

#include <accel/graph/node.hpp>

#include <node/accel/context.hpp>

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/sort/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] rund::kernel::SortDesc
SortDescFor(const rund::AccelGraphNode &node) noexcept;

[[nodiscard]] rund::kernel::CompactDesc
CompactDescFor(const rund::AccelGraphNode &node) noexcept;

[[nodiscard]] rund::kernel::ScatterDesc
ScatterDescFor(const rund::AccelGraphNode &node) noexcept;

} // namespace rund::node::accel::detail
