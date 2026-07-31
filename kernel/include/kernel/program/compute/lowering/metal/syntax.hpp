#pragma once

#include <kernel/program/compute/lowering/names.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr const char *
MetalType(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "long" : "int";
}

[[nodiscard]] constexpr const char *
MetalUnsignedType(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "ulong" : "uint";
}

[[nodiscard]] constexpr const char *
MetalLoadFunction(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "LoadI64" : "LoadI32";
}

[[nodiscard]] constexpr const char *
MetalParamLoadFunction(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "LoadParamI64" : "LoadParamI32";
}

[[nodiscard]] constexpr const char *
MetalStoreFunction(const ComputeScalar scalar) noexcept {
  return scalar == ComputeScalar::Lane64 ? "StoreI64" : "StoreI32";
}

[[nodiscard]] constexpr ComputeScalar
MetalStoreScalar(const u32 element_bytes) noexcept {
  return element_bytes == sizeof(u64) ? ComputeScalar::Lane64
                                      : ComputeScalar::Lane32;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
