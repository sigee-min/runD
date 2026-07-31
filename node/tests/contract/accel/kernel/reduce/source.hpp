#pragma once

[[nodiscard]] bool SignedReduceSourcesCarryDomainOrder() {
  const std::string metal = rund::node::accel::detail::MetalReduceSource(
      rund::kernel::ReduceOp::Min, 64u, rund::kernel::ComputeDomain::I32);
  if (metal.find("min(int(sums[tid]), int(rhs))") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = rund::node::accel::detail::VulkanReduceSource(
      rund::kernel::ReduceOp::Min, rund::kernel::ReduceElement::U32, 64u,
      rund::kernel::ComputeDomain::I32);
  if (vulkan.find("min(int64_t(int(uint(sums[tid])))") == std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool WideReduceSourcesCarryFixedHierarchy() {
  const std::string metal = rund::node::accel::detail::MetalReduceSource(
      rund::kernel::ReduceOp::Sum, 256u, rund::kernel::ComputeDomain::I64);
  if (metal.find("device RundWide* partial [[buffer(1)]]") ==
          std::string::npos ||
      metal.find(
          "struct RundWide { RUND_REDUCE_U64 lo; RUND_REDUCE_U64 hi; }") ==
          std::string::npos ||
      metal.find("params.grid_size * ulong(RUND_REDUCE_BLOCK_SIZE)") ==
          std::string::npos ||
      metal.find("partial[ulong(group)] = acc") == std::string::npos) {
    return false;
  }
  const std::string metal_u32 = rund::node::accel::detail::MetalReduceSource(
      rund::kernel::ReduceOp::Sum, 256u, rund::kernel::ComputeDomain::U32);
  if (metal_u32.find("items < RUND_REDUCE_NARROW_CHUNK") == std::string::npos ||
      metal_u32.find(
          "narrow = rund_pair_add(narrow, uint(input[index]), 0u)") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string vulkan = rund::node::accel::detail::VulkanReduceSource(
      rund::kernel::ReduceOp::CountNonzero, rund::kernel::ReduceElement::U64,
      256u, rund::kernel::ComputeDomain::U64);
  if (vulkan.find("uint64_t grid_size") == std::string::npos ||
      vulkan.find("const uint64_t stride = params.grid_size") ==
          std::string::npos ||
      vulkan.find("partial_values[word + 1u] = acc.hi") == std::string::npos) {
    return false;
  }
  const std::string vulkan_u32 = rund::node::accel::detail::VulkanReduceSource(
      rund::kernel::ReduceOp::Sum, rund::kernel::ReduceElement::U32, 256u,
      rund::kernel::ComputeDomain::U32);
  if (vulkan_u32.find("items < RUND_REDUCE_NARROW_CHUNK") ==
          std::string::npos ||
      vulkan_u32.find(
          "narrow = rund_pair_add(narrow, input_values[uint(index)], 0u)") ==
          std::string::npos) {
    return false;
  }
#endif
  return true;
}
