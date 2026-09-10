#include "local.hpp"

namespace rund::kernel::fusion_detail {
namespace {

[[nodiscard]] FusionHash MixPolicy(FusionHash hash,
                                   const FusionPolicy &policy) noexcept {
  hash = Mix(hash, policy.node_count);
  for (u64 index = 0u; index < policy.node_count; ++index) {
    hash = Mix(hash, index);
    hash = Mix(hash, policy.nodes[index].direct_read_mask);
    hash = Mix(hash, policy.nodes[index].supported ? 1u : 0u);
    hash = Mix(hash, policy.nodes[index].writes_visible ? 1u : 0u);
    hash = Mix(hash, policy.nodes[index].binding_count);
    hash = Mix(hash, policy.nodes[index].ir_node_count);
  }
  return hash;
}

} // namespace

FusionHash Mix(const FusionHash hash, const u64 value) noexcept {
  const graph_detail::GraphHash mixed = graph_detail::Mix(
      graph_detail::GraphHash{.hi = hash.hi, .lo = hash.lo}, value);
  return FusionHash{.hi = mixed.hi, .lo = mixed.lo};
}

FusionHash FusedOutputId(const GraphCheck &input, const FusionPolicy &policy,
                         const u64 original_node_count,
                         const u64 fused_node_count,
                         const u64 rejected_edge_count,
                         const FusionHash boundary_decisions) noexcept {
  if (original_node_count == fused_node_count && rejected_edge_count == 0u) {
    return FusionHash{.hi = input.graph_id_hi, .lo = input.graph_id_lo};
  }
  FusionHash hash{.hi = input.graph_id_hi, .lo = input.graph_id_lo};
  hash = Mix(hash, 0x667573696f6e5f76ull);
  hash = MixPolicy(hash, policy);
  hash = Mix(hash, original_node_count);
  hash = Mix(hash, fused_node_count);
  hash = Mix(hash, rejected_edge_count);
  hash = Mix(hash, boundary_decisions.hi);
  hash = Mix(hash, boundary_decisions.lo);
  return hash;
}

} // namespace rund::kernel::fusion_detail
