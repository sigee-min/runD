#include <kernel/program/compute/ir.hpp>

namespace rund::kernel::compute_ir_detail {

namespace {

constexpr u64 kFnvOffset = 14695981039346656037ull;
constexpr u64 kFnvPrime = 1099511628211ull;
constexpr u64 kSecondOffset = 10957381669161133421ull;
constexpr u64 kSecondPrime = 14029467366897019727ull;

}  // namespace

ComputeIrHash HashComputeIrCanonicalBytes(const u8* const bytes,
                                  const u64 size) noexcept {
  u64 hi = kFnvOffset;
  u64 lo = kSecondOffset;

  for (u64 index = 0u; index < size; ++index) {
    const u64 byte = bytes != nullptr ? static_cast<u64>(bytes[index]) : 0u;
    hi ^= byte;
    hi *= kFnvPrime;

    lo ^= byte + 0x9e3779b97f4a7c15ull + (lo << 6u) + (lo >> 2u);
    lo *= kSecondPrime;
  }

  if (hi == 0u && lo == 0u) {
    lo = kFnvPrime;
  }
  return ComputeIrHash{.hi = hi, .lo = lo};
}

}  // namespace rund::kernel::compute_ir_detail
