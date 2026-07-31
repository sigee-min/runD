#pragma once

#include "fixed.hpp"
#include "local.hpp"

namespace node_accel_contract::cpu::stat {

template <typename T> [[nodiscard]] T Abs(const T value) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAbs(value);
  } else {
    return rund::math32::detail::ScalarAbs(value);
  }
}

template <typename T> [[nodiscard]] T AddSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAddSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarAddSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T AddWrap(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAddWrap(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarAddWrap(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T DivFixed(const T lhs, const T rhs) {
  return fixed::DivNearestEven(lhs, rhs);
}

template <typename T> [[nodiscard]] T MulFixed(const T lhs, const T rhs) {
  return fixed::QuantizeProduct(lhs, rhs);
}

template <typename T> [[nodiscard]] T Sqrt(const T value) {
  return fixed::SqrtFloor(value);
}

template <typename T> [[nodiscard]] T SubSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarSubSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarSubSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] constexpr T Half() noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return 0x4000000000000000ll;
  } else {
    return 0x40000000;
  }
}

template <typename T> [[nodiscard]] constexpr T Third() noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return 3074457345618258602ll;
  } else {
    return 715827882;
  }
}

template <typename T> [[nodiscard]] constexpr T Quarter() noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return 0x2000000000000000ll;
  } else {
    return 0x20000000;
  }
}

template <typename T> [[nodiscard]] T Mean(const T lhs, const T rhs) {
  return AddSat(MulFixed(lhs, Half<T>()), MulFixed(rhs, Half<T>()));
}

template <typename T> [[nodiscard]] T Mean(const T a, const T b, const T c) {
  return AddSat(AddSat(MulFixed(a, Third<T>()), MulFixed(b, Third<T>())),
                MulFixed(c, Third<T>()));
}

template <typename T>
[[nodiscard]] T Mean(const T a, const T b, const T c, const T d) {
  return AddSat(AddSat(MulFixed(a, Quarter<T>()), MulFixed(b, Quarter<T>())),
                AddSat(MulFixed(c, Quarter<T>()), MulFixed(d, Quarter<T>())));
}

template <typename T> [[nodiscard]] T Centered(const T value, const T center) {
  return SubSat(value, center);
}

enum class CenteredOp { Abs, Squared };
template <typename T>
[[nodiscard]] T Centered(const CenteredOp op, const T value, const T center) {
  const T delta = Centered(value, center);
  return op == CenteredOp::Abs ? Abs(delta) : MulFixed(delta, delta);
}

template <typename T> [[nodiscard]] T Var(const T lhs, const T rhs) {
  const T mean = Mean(lhs, rhs);
  return Mean(Centered(CenteredOp::Squared, lhs, mean), Centered(CenteredOp::Squared, rhs, mean));
}

template <typename T> [[nodiscard]] T Var(const T a, const T b, const T c) {
  const T mean = Mean(a, b, c);
  return Mean(Centered(CenteredOp::Squared, a, mean), Centered(CenteredOp::Squared, b, mean), Centered(CenteredOp::Squared, c, mean));
}

template <typename T>
[[nodiscard]] T Var(const T a, const T b, const T c, const T d) {
  const T mean = Mean(a, b, c, d);
  return Mean(Centered(CenteredOp::Squared, a, mean),
              Centered(CenteredOp::Squared, b, mean),
              Centered(CenteredOp::Squared, c, mean),
              Centered(CenteredOp::Squared, d, mean));
}

template <typename T> [[nodiscard]] T Rms(const T lhs, const T rhs) {
  return Sqrt(Mean(MulFixed(lhs, lhs), MulFixed(rhs, rhs)));
}

template <typename T> [[nodiscard]] T Rms(const T a, const T b, const T c) {
  return Sqrt(Mean(MulFixed(a, a), MulFixed(b, b), MulFixed(c, c)));
}

template <typename T>
[[nodiscard]] T Rms(const T a, const T b, const T c, const T d) {
  return Sqrt(Mean(MulFixed(a, a), MulFixed(b, b), MulFixed(c, c),
                    MulFixed(d, d)));
}

template <typename T>
[[nodiscard]] T Cov(const T x0, const T x1, const T y0, const T y1) {
  const T mx = Mean(x0, x1);
  const T my = Mean(y0, y1);
  return Mean(MulFixed(Centered(x0, mx), Centered(y0, my)),
               MulFixed(Centered(x1, mx), Centered(y1, my)));
}

template <typename T>
[[nodiscard]] T Cov(const T x0, const T x1, const T x2, const T y0,
                     const T y1, const T y2) {
  const T mx = Mean(x0, x1, x2);
  const T my = Mean(y0, y1, y2);
  return Mean(MulFixed(Centered(x0, mx), Centered(y0, my)),
               MulFixed(Centered(x1, mx), Centered(y1, my)),
               MulFixed(Centered(x2, mx), Centered(y2, my)));
}

template <typename T>
[[nodiscard]] T Cov(const T x0, const T x1, const T x2, const T x3,
                     const T y0, const T y1, const T y2, const T y3) {
  const T mx = Mean(x0, x1, x2, x3);
  const T my = Mean(y0, y1, y2, y3);
  return Mean(MulFixed(Centered(x0, mx), Centered(y0, my)),
               MulFixed(Centered(x1, mx), Centered(y1, my)),
               MulFixed(Centered(x2, mx), Centered(y2, my)),
               MulFixed(Centered(x3, mx), Centered(y3, my)));
}

template <typename T>
[[nodiscard]] T Corr(const T x0, const T x1, const T y0, const T y1) {
  const T denom = Sqrt(MulFixed(Var(x0, x1), Var(y0, y1)));
  return denom == T{0} ? T{0} : DivFixed(Cov(x0, x1, y0, y1), denom);
}

template <typename T>
[[nodiscard]] T Corr(const T x0, const T x1, const T x2, const T y0,
                      const T y1, const T y2) {
  const T denom = Sqrt(MulFixed(Var(x0, x1, x2), Var(y0, y1, y2)));
  return denom == T{0} ? T{0} : DivFixed(Cov(x0, x1, x2, y0, y1, y2), denom);
}

template <typename T>
[[nodiscard]] T Corr(const T x0, const T x1, const T x2, const T x3,
                      const T y0, const T y1, const T y2, const T y3) {
  const T denom =
      Sqrt(MulFixed(Var(x0, x1, x2, x3), Var(y0, y1, y2, y3)));
  return denom == T{0}
             ? T{0}
             : DivFixed(Cov(x0, x1, x2, x3, y0, y1, y2, y3), denom);
}

} // namespace node_accel_contract::cpu::stat
