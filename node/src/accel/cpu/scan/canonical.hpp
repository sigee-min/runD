#pragma once

#include <accel/check.hpp>

#include <kernel/program/compute/model.hpp>

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::node::accel::detail {

template <std::unsigned_integral Word>
using ScanSigned = std::make_signed_t<Word>;

template <std::unsigned_integral Word>
using ScanWideSigned = std::conditional_t<sizeof(Word) == sizeof(std::uint32_t),
                                          std::int64_t, __int128_t>;

template <std::unsigned_integral Word>
[[nodiscard]] constexpr bool
SignedPrefixFits(const ScanWideSigned<Word> prefix) noexcept {
  using Signed = ScanSigned<Word>;
  return prefix >= static_cast<ScanWideSigned<Word>>(
                       std::numeric_limits<Signed>::min()) &&
         prefix <= static_cast<ScanWideSigned<Word>>(
                       std::numeric_limits<Signed>::max());
}

template <std::unsigned_integral Word>
[[nodiscard]] constexpr bool
StoreSignedPrefix(const ScanWideSigned<Word> prefix,
                  Word *const output) noexcept {
  using Signed = ScanSigned<Word>;
  if (!SignedPrefixFits<Word>(prefix)) {
    return false;
  }
  *output = std::bit_cast<Word>(static_cast<Signed>(prefix));
  return true;
}

template <std::unsigned_integral Word>
[[nodiscard]] inline rund::AccelCheck
CanonicalSignedScan(const Word *const input, Word *const output,
                    const rund::kernel::u64 element_count,
                    const bool inclusive) noexcept {
  if (element_count == 0u) {
    return rund::AccelCheck{false, "compute_scan_count_zero"};
  }
  if (input == nullptr || output == nullptr) {
    return rund::AccelCheck{false, "compute_scan_buffer_invalid"};
  }
  ScanWideSigned<Word> running = 0;
  for (rund::kernel::u64 index = 0u; index < element_count; ++index) {
    if (!inclusive && !StoreSignedPrefix(running, output + index)) {
      return rund::AccelCheck{false, "compute_scan_sum_overflow"};
    }
    running += static_cast<ScanWideSigned<Word>>(
        std::bit_cast<ScanSigned<Word>>(input[index]));
    if (inclusive && !StoreSignedPrefix(running, output + index)) {
      return rund::AccelCheck{false, "compute_scan_sum_overflow"};
    }
  }
  if (!inclusive && !SignedPrefixFits<Word>(running)) {
    return rund::AccelCheck{false, "compute_scan_sum_overflow"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
