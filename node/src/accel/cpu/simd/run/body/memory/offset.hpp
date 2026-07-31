#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] std::size_t ByteOffset(const u64 tile,
                                     const std::size_t stride) noexcept {
  return static_cast<std::size_t>(tile * static_cast<u64>(stride));
}

}  // namespace
}  // namespace rund::node::accel::cpu_simd_detail
