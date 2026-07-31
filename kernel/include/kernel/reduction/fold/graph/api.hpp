#pragma once

#include <kernel/reduction/fold/graph/result.hpp>
#include <kernel/reduction/fold/graph/state.hpp>
#include <kernel/reduction/fold/slots.hpp>
#include <kernel/schedule/planner/policy.hpp>

#include <span>

namespace rund::kernel {

u32 FoldGraphFixedBinaryTreeEdgeCount(u32 partition_count);
u32 FoldGraphFixedBinaryTreeNodeCount(u32 partition_count);
u32 FoldGraphFixedBinaryTreeScratchSlotCount(u32 partition_count);
u32 FoldGraphScratchSlotCount(FoldGraphView graph);
FoldGraphValidationResult ValidateFoldGraph(FoldGraphView graph);
bool EnsureFoldGraphScratch(FoldSlots& scratch,
                            FoldGraphView graph,
                            AllocationPolicy allocation);
FoldResult FoldGraphReduce(FoldGraphView graph,
                           std::span<const u64> partition_values,
                           std::span<u64> scratch_slots);
FoldResult FoldGraphReduce(FoldGraphView graph,
                           std::span<const u64> partition_values,
                           FoldSlots& scratch,
                           AllocationPolicy allocation);
void ResetFoldGraph(FoldGraph& graph);
bool ReserveFoldGraph(FoldGraph& graph, u32 partition_capacity);
FoldGraphBuild BuildFoldGraph(FoldGraph& graph,
                              u32 partition_count,
                              FoldOperation operation,
                              AllocationPolicy allocation);
FoldGraphBuild BuildFoldGraph(FoldGraph& graph,
                              u32 partition_count,
                              FoldOperation operation,
                              StrictFloatReductionPolicy strict_float_reduction,
                              AllocationPolicy allocation);
FoldGraphView ViewFoldGraph(const FoldGraph& graph);

} // namespace rund::kernel
