#pragma once

#include <kernel/reduction/fold/graph/result.hpp>
#include <kernel/reduction/fold/graph/state.hpp>
#include <kernel/reduction/fold/primitive.hpp>
#include <kernel/schedule/planner/policy.hpp>

namespace rund::kernel::reduction::graph {

FoldGraphValidationResult ValidateBuiltFixedBinaryTreeGraph(
    const FoldGraph &graph, FoldOperation operation,
    const FoldPrimitiveSpec &primitive, u32 partition_count, u32 required_nodes,
    u32 required_edges, u32 required_scratch_slots);

} // namespace rund::kernel::reduction::graph
