#pragma once

#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/value.hpp>

#include "../context/internal.hpp"
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rund::node::accel::detail {

struct KernelTokenMint {
  std::shared_ptr<void> owner{};
  std::uint64_t kernel_id = 0u;
};

[[nodiscard]] KernelTokenMint MintKernelToken(
    const ContextAdmission &admission,
    const rund::kernel::GraphCheck &graph_check,
    std::vector<KernelExecutionStep> steps,
    std::vector<std::uint8_t> required_barriers,
    std::uint64_t removed_dispatch_count,
    std::vector<rund::kernel::BufferRole> graph_roles,
    std::vector<rund::AccelBufferDesc> graph_shapes,
    std::vector<rund::GraphBufferVisibility> graph_visibilities,
    std::vector<std::uint64_t> graph_alias_representatives,
    std::vector<SourceStep> graph_binding_sources,
    std::vector<std::uint64_t> graph_reset_bindings,
    rund::kernel::ComputeScalar scalar, rund::kernel::ComputeDomain domain,
    std::uint64_t original_operation_count, std::uint64_t fused_operation_count,
    std::uint64_t fusion_rejection_count, const char *fusion_reason);

// Authenticates the opaque public owner through its sealed control-block
// capability, then measures the immutable token allocation tree without
// allocating. Shared context ownership belongs to its own owner and is
// intentionally outside this Program-retained result.
[[nodiscard]] std::optional<std::uint64_t>
MeasureKernelTokenRetainedMemory(const rund::AccelKernel &kernel) noexcept;

} // namespace rund::node::accel::detail
