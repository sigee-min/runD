#pragma once

#include <accel/context/value.hpp>
#include <accel/graph/node.hpp>

#include "../../context/shared.hpp"
#include "../admission.hpp"
#include "../compile.hpp"
namespace rund::node::accel::detail {

void PopulateBaseCompileData(const rund::AccelGraphNode &node,
                             GraphCompileNode &compile_data) noexcept;

[[nodiscard]] const char *
AppendGraphBufferRefs(const ContextAdmission &admission,
                      const rund::AccelGraphNode &node, SourceStep source,
                      GraphCompileState &state, GraphCompileNode &compile_data,
                      std::vector<rund::kernel::GraphBufferRef> &buffers);

void AppendKernelGraphNode(
    const rund::AccelGraphNode &node, const GraphCompileNode &compile_data,
    const std::vector<rund::kernel::GraphBufferRef> &buffers,
    GraphCompileState &state);

} // namespace rund::node::accel::detail
