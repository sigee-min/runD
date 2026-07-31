#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::cpu_simd_detail {
namespace {

using ValueVec = RUND_CPU_SIMD_VEC;
using WideScalar = __int128_t;
using StoredScalar = RUND_CPU_SIMD_SCALAR;
using StoredBits = RUND_CPU_SIMD_BITS_SCALAR;
inline constexpr std::size_t kValueLaneCount = RUND_CPU_SIMD_LANES;

class Values {
public:
  explicit Values(ValueVec *const data, WideScalar *const wide,
                  std::uint8_t *const wide_valid) noexcept
      : data_(data), wide_(wide), wide_valid_(wide_valid) {}
  [[nodiscard]] ValueVec &operator[](const std::size_t index) noexcept {
    return data_[index];
  }

  [[nodiscard]] const ValueVec &
  operator[](const std::size_t index) const noexcept {
    return data_[index];
  }

  void invalidate(const std::size_t index) noexcept { wide_valid_[index] = 0u; }

  [[nodiscard]] WideScalar wide(const std::size_t index,
                                const std::size_t lane) noexcept {
    materialize(index);
    return wide_[index * kValueLaneCount + lane];
  }

  void set_wide(const std::size_t index,
                const std::array<WideScalar, kValueLaneCount> &lanes) noexcept {
    std::array<StoredScalar, kValueLaneCount> stored{};
    for (std::size_t lane = 0u; lane < kValueLaneCount; ++lane) {
      wide_[index * kValueLaneCount + lane] = lanes[lane];
      const StoredBits bits =
          static_cast<StoredBits>(static_cast<__uint128_t>(lanes[lane]));
      stored[lane] = std::bit_cast<StoredScalar>(bits);
    }
    data_[index] = RUND_CPU_SIMD_LOAD(stored.data());
    wide_valid_[index] = 1u;
  }
  void fail(const char *const reason) noexcept {
    reason_ = reason_ == nullptr ? reason : reason_;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return reason_ == nullptr;
  }

  [[nodiscard]] const char *reason() const noexcept {
    return reason_ ? reason_ : "ok";
  }

private:
  void materialize(const std::size_t index) noexcept {
    if (wide_valid_[index] != 0u) {
      return;
    }
    std::array<StoredScalar, kValueLaneCount> stored{};
    RUND_CPU_SIMD_STORE(stored.data(), data_[index]);
    for (std::size_t lane = 0u; lane < kValueLaneCount; ++lane) {
      wide_[index * kValueLaneCount + lane] =
          static_cast<WideScalar>(stored[lane]);
    }
    wide_valid_[index] = 1u;
  }

  ValueVec *data_ = nullptr;
  WideScalar *wide_ = nullptr;
  std::uint8_t *wide_valid_ = nullptr;
  const char *reason_ = nullptr;
};

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
