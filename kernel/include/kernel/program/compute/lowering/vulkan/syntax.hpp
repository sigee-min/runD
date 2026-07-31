#pragma once

#include <kernel/program/compute/lowering/layout.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr const char *
VulkanType(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "uint64_t" : "uint";
}

[[nodiscard]] constexpr const char *
VulkanLoadPrefix(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "LoadI64" : "LoadI32";
}

[[nodiscard]] constexpr const char *
VulkanParamLoadFunction(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "LoadParamI64" : "LoadParamI32";
}

[[nodiscard]] constexpr const char *
VulkanStorePrefix(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "StoreI64" : "StoreI32";
}

[[nodiscard]] constexpr ComputeScalar
VulkanStoreScalar(const u32 element_bytes) noexcept {
  return element_bytes == sizeof(u64) ? ComputeScalar::Lane64
                                      : ComputeScalar::Lane32;
}

[[nodiscard]] inline std::string VulkanDataSymbol(const BindingLayout &layout) {
  return layout.symbol + "_data";
}

[[nodiscard]] inline std::string
VulkanLoadFunctionName(const ComputeScalar scalar,
                       const BindingLayout &layout) {
  std::string name = VulkanLoadPrefix(scalar);
  name += "_";
  name += layout.symbol;
  return name;
}

[[nodiscard]] inline std::string
VulkanStoreFunctionName(const ComputeScalar scalar,
                        const BindingLayout &layout) {
  std::string name = VulkanStorePrefix(scalar);
  name += "_";
  name += layout.symbol;
  return name;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
