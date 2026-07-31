#pragma once

#include <accel/graph/value.hpp>
#include <kernel/program/compute/graph/signature.hpp>

namespace rund::accel_graph_factory_detail {

[[nodiscard]] inline AccelGraphNode
MapNode(const rund::kernel::ComputeIR &ir,
        const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
        const std::uint64_t element_count) {
  return AccelGraphNode{
      .ir = &ir,
      .buffers = refs,
      .buffer_count = ref_count,
      .kind = rund::kernel::NodeKind::Map,
      .element_count = element_count,
      .signature = rund::kernel::BuildMapGraphSignature(
          ir, rund::kernel::ComputeApi::Cpu),
  };
}

template <typename Hash>
[[nodiscard]] inline AccelGraphNode
PrimitiveNode(const AccelGraphBufferRef *const refs,
              const std::uint64_t ref_count, const rund::kernel::NodeKind kind,
              const Hash &hash, const std::uint64_t element_count) noexcept {
  return AccelGraphNode{
      .buffers = refs,
      .buffer_count = ref_count,
      .kind = kind,
      .primitive_hash_hi = hash.hi,
      .primitive_hash_lo = hash.lo,
      .element_count = element_count,
  };
}

} // namespace rund::accel_graph_factory_detail
