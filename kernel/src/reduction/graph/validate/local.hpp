#pragma once

#include "../local.hpp"

namespace rund::kernel::reduction::graph::validate_detail {

FoldGraphValidationResult FailBuiltGraphValidation(const char* reason,
                                                   u32 scratch_slot_count = 0u);

FoldGraphValidationResult ValidateFixedGraphShape(const FoldGraph& graph,
                                                  const FoldPrimitiveSpec& primitive,
                                                  u32 partition_count,
                                                  u32 required_nodes,
                                                  u32 required_edges,
                                                  u32 required_scratch_slots);

FoldGraphValidationResult ValidateFixedGraphPartitionNodes(const FoldGraph& graph,
                                                           FoldOperation operation,
                                                           const FoldPrimitiveSpec& primitive,
                                                           u32 partition_count,
                                                           u32 required_scratch_slots);

FoldGraphValidationResult ValidateFixedGraphReductionEdges(const FoldGraph& graph,
                                                           FoldOperation operation,
                                                           const FoldPrimitiveSpec& primitive,
                                                           u32 partition_count,
                                                           u32 required_edges,
                                                           u32 required_scratch_slots);

} // namespace rund::kernel::reduction::graph::validate_detail
