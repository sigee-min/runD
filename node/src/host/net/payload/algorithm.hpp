#pragma once

#include "../native/result.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::net::payload_detail {

template <typename Identity>
[[nodiscard]] ::rund::StableHash
Native(const node::NativeCallResult &native, const std::byte *const data,
       const std::uint64_t requested, Identity &identity) noexcept {
  if (native.value < 0) {
    return {};
  }
  return identity.Hash(
      data, static_cast<std::size_t>(CompletedByteCount(native, requested)));
}

template <typename Slice, typename Identity>
[[nodiscard]] ::rund::StableHash Prefix(const std::span<const Slice> slices,
                                        const std::uint64_t completed_bytes,
                                        Identity &identity) noexcept {
  if (completed_bytes == 0u) {
    return identity.Hash(nullptr, 0u);
  }

  std::uint64_t remaining = completed_bytes;
  const std::byte *first = nullptr;
  std::size_t first_size = 0u;
  bool have_first = false;
  for (std::size_t index = 0u; index < slices.size(); ++index) {
    const Slice slice = slices[index];
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, slice.size));
    if constexpr (requires { identity.Visit(slice, count); }) {
      identity.Visit(slice, count);
    }
    remaining -= count;
    if (count == 0u) {
      if (remaining == 0u) {
        break;
      }
      continue;
    }
    if (!have_first) {
      first = slice.data;
      first_size = count;
      have_first = true;
      if (remaining == 0u) {
        break;
      }
      continue;
    }

    auto hash = identity.Begin();
    hash.Append({first, first_size});
    hash.Append({slice.data, count});
    for (++index; remaining != 0u && index < slices.size(); ++index) {
      const Slice tail = slices[index];
      const std::size_t tail_count = static_cast<std::size_t>(
          std::min<std::uint64_t>(remaining, tail.size));
      if constexpr (requires { identity.Visit(tail, tail_count); }) {
        identity.Visit(tail, tail_count);
      }
      hash.Append({tail.data, tail_count});
      remaining -= tail_count;
    }
    return remaining == 0u
               ? hash.Finish()
               : identity.Hash(nullptr,
                               static_cast<std::size_t>(completed_bytes));
  }
  if (remaining != 0u) {
    return identity.Hash(nullptr, static_cast<std::size_t>(completed_bytes));
  }
  return identity.Hash(first, first_size);
}

} // namespace rund::net::payload_detail
