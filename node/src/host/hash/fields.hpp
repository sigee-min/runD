#pragma once

#include <rund/host/hash.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::node::host_detail {

inline constexpr std::uint64_t kStableHashOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kStableHashPrime = 1099511628211ull;

class StableHashState {
public:
  constexpr StableHashState() noexcept = default;

  constexpr void Mix(const std::uint64_t value) noexcept {
    value_ ^= value;
    value_ *= kStableHashPrime;
  }

  constexpr void MixBytes(const std::span<const std::byte> bytes) noexcept {
    std::size_t index = 0u;
    while (bytes.size() - index >= 8u) {
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 1u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 2u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 3u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 4u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 5u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 6u])) *
               kStableHashPrime;
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index + 7u])) *
               kStableHashPrime;
      index += 8u;
    }
    while (index < bytes.size()) {
      value_ = (value_ ^ std::to_integer<std::uint8_t>(bytes[index])) *
               kStableHashPrime;
      ++index;
    }
  }

  [[nodiscard]] constexpr ::rund::StableHash Finish() const noexcept {
    return ::rund::StableHash{.value = value_};
  }

private:
  std::uint64_t value_ = kStableHashOffset;
};

// Hashes an already-canonicalized structured field sequence without
// materializing it. The declared byte count is part of this structured
// framing and every appended integer is emitted in explicit little-endian
// byte order. Raw host payload bytes use hash/bytes.hpp instead.
class CanonicalByteHasher {
public:
  explicit constexpr CanonicalByteHasher(
      const std::uint64_t byte_count) noexcept {
    state_.Mix(byte_count);
  }

  constexpr void AppendByte(const std::byte value) noexcept {
    state_.Mix(std::to_integer<std::uint8_t>(value));
  }

  constexpr void AppendU32Le(const std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
      AppendByte(static_cast<std::byte>((value >> shift) & 0xffu));
    }
  }

  constexpr void AppendU64Le(const std::uint64_t value) noexcept {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
      AppendByte(static_cast<std::byte>((value >> shift) & 0xffu));
    }
  }

  constexpr void AppendBytes(const std::span<const std::byte> bytes) noexcept {
    state_.MixBytes(bytes);
  }

  constexpr void
  AppendLengthPrefixedBytes(const std::span<const std::byte> bytes) noexcept {
    AppendU64Le(static_cast<std::uint64_t>(bytes.size()));
    AppendBytes(bytes);
  }

  constexpr void
  AppendLengthPrefixedString(const std::string_view text) noexcept {
    AppendU64Le(static_cast<std::uint64_t>(text.size()));
    for (const char value : text) {
      AppendByte(static_cast<std::byte>(static_cast<unsigned char>(value)));
    }
  }

  [[nodiscard]] constexpr ::rund::StableHash Finish() const noexcept {
    return state_.Finish();
  }

private:
  StableHashState state_{};
};

} // namespace rund::node::host_detail
