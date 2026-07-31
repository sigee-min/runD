#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

inline constexpr u32 kMaxComputeBindingCount = 64u;
inline constexpr u32 kMaxComputeNodeCount = 1024u;

// A Program graph is an ordered schedule of already-bounded Compute nodes.
// Its schedule envelope is intentionally wider than one ComputeIR so a large
// domain graph does not have to inflate the per-kernel IR or split execution
// authority across Programs.
inline constexpr u64 kMaxGraphNodeCount = 16u * kMaxComputeNodeCount;
inline constexpr u64 kMaxGraphBuffersPerNode = kMaxComputeBindingCount;
inline constexpr u64 kMaxGraphOutputCount = 64u;

// Every admitted logical resource must occur in at least one bounded node
// reference or in the bounded public output set. This is the complete
// representable value envelope, not an independently tuned magic number.
inline constexpr u64 kMaxGraphValueCount =
    kMaxGraphNodeCount * kMaxGraphBuffersPerNode + kMaxGraphOutputCount;

} // namespace rund::kernel
