#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::node::hash_detail {

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
inline constexpr std::uint64_t kFnvStandardOffset = 14695981039346656037ull;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ull;

class Fnv final {
public:
  explicit constexpr Fnv(const std::uint64_t seed = kFnvOffset) noexcept
      : value_{seed} {}

  constexpr void Byte(const std::uint8_t value) noexcept {
    value_ = (value_ ^ value) * kFnvPrime;
  }

  template <class Value>
  constexpr void Number(const Value value) noexcept {
    using Unsigned = std::make_unsigned_t<Value>;
    const Unsigned bits = static_cast<Unsigned>(value);
    for (std::size_t byte = 0u; byte < sizeof(Value); ++byte) {
      Byte(static_cast<std::uint8_t>(
          bits >> static_cast<unsigned>(byte * 8u)));
    }
  }

  constexpr void Bytes(const std::uint8_t *const source,
                       const std::size_t count) noexcept {
    for (std::size_t index = 0u; index < count; ++index) {
      Byte(source[index]);
    }
  }

  [[nodiscard]] constexpr std::uint64_t Finish() const noexcept {
    return value_;
  }

private:
  std::uint64_t value_;
};

static_assert([]() constexpr {
  Fnv hash{};
  hash.Byte(static_cast<std::uint8_t>('a'));
  hash.Byte(static_cast<std::uint8_t>('b'));
  hash.Byte(static_cast<std::uint8_t>('c'));
  return hash.Finish();
}() == 0xe16801510db89efdull);

static_assert([]() constexpr {
  Fnv hash{kFnvStandardOffset};
  hash.Byte(static_cast<std::uint8_t>('a'));
  hash.Byte(static_cast<std::uint8_t>('b'));
  hash.Byte(static_cast<std::uint8_t>('c'));
  return hash.Finish();
}() == 0xe71fa2190541574bull);

[[nodiscard]] inline std::uint64_t
HashBytes(const void *const source, const std::size_t bytes) noexcept {
  const auto *input = static_cast<const std::uint8_t *>(source);
  Fnv hash{};
  hash.Bytes(input, bytes);
  return hash.Finish();
}

[[nodiscard]] inline std::uint64_t
CopyHash(const void *const source, void *const target,
         const std::size_t bytes) noexcept {
  const auto *input = static_cast<const std::uint8_t *>(source);
  auto *output = static_cast<std::uint8_t *>(target);
  Fnv hash{};
  for (std::size_t index = 0u; index < bytes; ++index) {
    const std::uint8_t value = input[index];
    output[index] = value;
    hash.Byte(value);
  }
  return hash.Finish();
}

[[nodiscard]] inline std::uint64_t ZeroHash(std::size_t bytes) noexcept {
  // For a zero byte, FNV-1a reduces to h' = h * prime modulo 2^64.
  std::uint64_t factor = 1u;
  std::uint64_t power = kFnvPrime;
  while (bytes != 0u) {
    if ((bytes & 1u) != 0u) {
      factor *= power;
    }
    power *= power;
    bytes >>= 1u;
  }
  return kFnvOffset * factor;
}

inline void MixU64(std::uint64_t &hash, const std::uint64_t value) noexcept {
  Fnv state{hash};
  state.Number(value);
  hash = state.Finish();
}

} // namespace rund::node::hash_detail
