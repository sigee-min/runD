#pragma once

#include <accel/graph/node.hpp>

#include "compile.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] const char *AdmitGraphNodePrimitive(
    const rund::AccelGraphNode &node, rund::kernel::ComputeApi api,
    GraphCompileNode &compile_data);

} // namespace rund::node::accel::detail
