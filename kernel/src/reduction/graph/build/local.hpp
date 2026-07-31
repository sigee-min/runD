#pragma once

#include "../local.hpp"

namespace rund::kernel::reduction::graph::build_detail {

struct BuildPlan {
  u32 partition_count = 0u;
  u32 required_edges = 0u;
  u32 required_nodes = 0u;
  u32 required_scratch_slots = 0u;
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldPrimitiveSpec primitive{};
  StrictFloatReductionPolicy strict_float_reduction{};
  AllocationPolicy allocation = AllocationPolicy::AllowGrowth;
};

FoldGraphBuild PrimitiveFailure(const BuildPlan& plan, const char* reason);
FoldGraphBuild CapacityFailure(const BuildPlan& plan, const char* reason);
FoldGraphBuild ValidationFailure(const FoldGraph& graph,
                                 const BuildPlan& plan,
                                 const FoldGraphValidationResult& validation);
FoldGraphBuild Success(const FoldGraph& graph,
                       const BuildPlan& plan,
                       const FoldGraphValidationResult& validation);
void AppendPartitionSlots(FoldGraph& graph, const BuildPlan& plan);
void AppendReductionTree(FoldGraph& graph, const BuildPlan& plan);
void PopulateFixedBinaryTreeGraph(FoldGraph& graph, const BuildPlan& plan);

} // namespace rund::kernel::reduction::graph::build_detail
