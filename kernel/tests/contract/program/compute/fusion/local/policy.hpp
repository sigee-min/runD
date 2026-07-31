#pragma once

#include "base.hpp"

namespace program_compute_contract::fusion_support {

struct TestOpHash {
  rund::kernel::u64 op_hash_hi = 0u;
  rund::kernel::u64 op_hash_lo = 0u;
};

constexpr TestOpHash kFirstOp{
    .op_hash_hi = 0x1020304050607080ull,
    .op_hash_lo = 0x8877665544332211ull,
};

constexpr TestOpHash kSecondOp{
    .op_hash_hi = 0x2122232425262728ull,
    .op_hash_lo = 0x9988776655443322ull,
};

constexpr TestOpHash kUnsupportedOp{
    .op_hash_hi = 0x3132333435363738ull,
    .op_hash_lo = 0xaabbccddeeff0011ull,
};

constexpr TestOpHash kThirdOp{
    .op_hash_hi = 0x4142434445464748ull,
    .op_hash_lo = 0x123456789abcdef0ull,
};

[[nodiscard]] constexpr rund::kernel::FusionNodePolicy
SupportedNode(const bool writes_visible = false,
              const rund::kernel::u32 binding_count = 2u,
              const rund::kernel::u32 ir_node_count = 2u) noexcept {
  return rund::kernel::FusionNodePolicy{
      .direct_read_mask = ~rund::kernel::u64{0u},
      .supported = true,
      .writes_visible = writes_visible,
      .binding_count = binding_count,
      .ir_node_count = ir_node_count,
  };
}

[[nodiscard]] inline rund::kernel::FusionNodePolicy
PolicyNode(const rund::kernel::ComputeIR &ir,
           const bool writes_visible = false) {
  auto input = rund::kernel::compute_lowering_detail::AdmitComputeInput(
      ir, rund::kernel::ComputeApi::Cpu);
  if (!input.ok) {
    return {};
  }
  return SupportedNode(
      writes_visible,
      static_cast<rund::kernel::u32>(input.parsed.bindings.size()),
      static_cast<rund::kernel::u32>(input.parsed.nodes.size()));
}

[[nodiscard]] inline rund::kernel::FusionPolicy
SupportedPolicy(const rund::kernel::FusionNodePolicy *const nodes,
                const rund::kernel::u64 node_count) noexcept {
  return rund::kernel::FusionPolicy{.nodes = nodes, .node_count = node_count};
}

} // namespace program_compute_contract::fusion_support
