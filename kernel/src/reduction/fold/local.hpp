#pragma once

#include <kernel/reduction/fold/graph/result.hpp>
#include <kernel/reduction/fold/graph/state.hpp>
#include <kernel/reduction/fold/primitive.hpp>

#include <span>

namespace rund::kernel::reduction::fold {

u64 FoldIdentityValue(FoldOperation operation);
u64 FoldPaddingValue(FoldOperation operation);
u64 CombineFoldValues(FoldOperation operation, u64 left, u64 right,
                      StrictFloatReductionPolicy policy);
u32 FixedBinaryTreeEdgeCount(u32 slot_count);
FoldGraphValidationResult
ValidateFoldGraphWithMarkers(FoldGraphView graph,
                             std::span<u64> defined_markers);

} // namespace rund::kernel::reduction::fold
