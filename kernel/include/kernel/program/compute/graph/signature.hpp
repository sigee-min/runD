#pragma once

#include <kernel/program/compute/graph/schema.hpp>

namespace rund::kernel {

struct CompactPlan;
struct ComputeIR;
struct ExecutionMetadata;
struct FactorPlan;
struct GatherPlan;
struct HistogramPlan;
struct MatrixPlan;
struct PartitionPlan;
struct ReducePlan;
struct ScanPlan;
struct ScatterPlan;
struct ScatterReducePlan;
struct SegmentedReducePlan;
struct SegmentedScanPlan;
struct SolvePlan;
struct SortPlan;
struct SpectrumPlan;
struct StencilPlan;
struct TransformPlan;

[[nodiscard]] GraphSignature BuildMapGraphSignature(const ComputeIR &ir,
                                                    ComputeApi api);

[[nodiscard]] GraphSignature
GraphSignatureFor(const ExecutionMetadata &metadata) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const ScanPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const SegmentedScanPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const SegmentedReducePlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const SortPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const CompactPlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const GatherPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const HistogramPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const PartitionPlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const ReducePlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const ScatterPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const ScatterReducePlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const StencilPlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const TransformPlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const MatrixPlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const FactorPlan &plan) noexcept;
[[nodiscard]] GraphSignature GraphSignatureFor(const SolvePlan &plan) noexcept;
[[nodiscard]] GraphSignature
GraphSignatureFor(const SpectrumPlan &plan) noexcept;

} // namespace rund::kernel
