#pragma once

#include <accel/graph/node.hpp>

#include "compile.hpp"
namespace rund::node::accel::detail {

struct ContextAdmission;

[[nodiscard]] const char *AdmitGraphNodePrimitive(
    const rund::AccelGraphNode &node, const ContextAdmission &admission,
    rund::kernel::ComputeDomain domain, GraphCompileNode &compile_data);

} // namespace rund::node::accel::detail
