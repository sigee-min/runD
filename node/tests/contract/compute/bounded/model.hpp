#pragma once

#include "local.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace rund_node_bounded_contract {

[[nodiscard]] inline std::uint64_t HashBytes(const void *const data,
                                             const std::size_t bytes) noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto *values = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = offset;
  for (std::size_t index = 0u; index < bytes; ++index) {
    hash ^= values[index];
    hash *= prime;
  }
  return hash;
}

inline void MixHash(std::uint64_t &hash, const std::uint64_t value) noexcept {
  constexpr std::uint64_t prime = 1099511628211ull;
  for (unsigned shift = 0u; shift < 64u; shift += 8u) {
    hash ^= (value >> shift) & 0xffu;
    hash *= prime;
  }
}

[[nodiscard]] inline std::uint64_t EmptyCompactHash() noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::array<std::uint32_t, 1u> zero_count{0u};
  std::uint64_t hash = offset;
  MixHash(hash, 2u);
  MixHash(hash, 0u);
  MixHash(hash, static_cast<std::uint64_t>(rund::compute::detail::Type::U32));
  MixHash(hash, 0u);
  MixHash(hash, HashBytes(nullptr, 0u));
  MixHash(hash, 1u);
  MixHash(hash, static_cast<std::uint64_t>(rund::compute::detail::Type::U32));
  MixHash(hash, 1u);
  MixHash(hash, HashBytes(zero_count.data(), sizeof(zero_count)));
  return hash;
}

template <class T> [[nodiscard]] constexpr T Small(const std::uint64_t value) {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(static_cast<std::int32_t>(value));
  } else if constexpr (std::same_as<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(static_cast<std::int64_t>(value));
  } else {
    return static_cast<T>(value);
  }
}

} // namespace rund_node_bounded_contract
