#pragma once

#include "../../local.hpp"

namespace rund::kernel::reduction::fold {

constexpr u64 kFoldGraphValidationMarkerTagMask = 0xC000000000000000ull;
constexpr u64 kFoldGraphValidationMarkerPayloadMask = ~kFoldGraphValidationMarkerTagMask;
constexpr u64 kFoldGraphValidationPartitionMarker = 0x4000000000000000ull;
constexpr u64 kFoldGraphValidationEdgeMarker = 0x8000000000000000ull;

FoldGraphValidationResult FailFoldGraphValidation(const char* reason,
                                                  u32 scratch_slot_count = 0u);
FoldGraphValidationResult ValidateFoldGraphShape(FoldGraphView graph);
FoldGraphValidationResult ValidateFoldGraphSlots(FoldGraphView graph,
                                                 std::span<u64> defined_markers,
                                                 u32 required_slots);
FoldGraphValidationResult ValidateFoldGraphEdges(FoldGraphView graph,
                                                 std::span<u64> defined_markers,
                                                 const FoldPrimitiveSpec& primitive,
                                                 u32 required_slots,
                                                 u32& final_slot);
FoldGraphValidationResult ValidateFoldGraphNodes(FoldGraphView graph,
                                                 std::span<const u64> defined_markers,
                                                 const FoldPrimitiveSpec& primitive,
                                                 u32 required_slots);

} // namespace rund::kernel::reduction::fold
