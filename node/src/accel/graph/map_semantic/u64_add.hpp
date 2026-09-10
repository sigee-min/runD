#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

// Recognizes the exact canonical left-associated U64 wrap-add chain emitted
// by Flow Map composition. Every IR node must belong to the one read/index,
// constant/add chain, and final value write; unused or reordered operations
// fail closed instead of becoming hidden fused semantics.
[[nodiscard]] bool ClassifyU64AddImmediateChain(
    const rund::kernel::ArtifactKey &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    std::uint64_t &immediate) noexcept;

} // namespace rund::node::accel::detail
