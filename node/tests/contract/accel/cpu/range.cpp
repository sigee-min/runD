#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Clamp(const T value, const T lo, const T hi) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarClamp(value, lo, hi);
  } else {
    return rund::math32::detail::ScalarClamp(value, lo, hi);
  }
}

template <typename T> [[nodiscard]] T SubSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarSubSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarSubSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T Min(const T lhs, const T rhs) {
  return lhs < rhs ? lhs : rhs;
}

template <typename T> [[nodiscard]] T Max(const T lhs, const T rhs) {
  return lhs > rhs ? lhs : rhs;
}

template <typename T> [[nodiscard]] T Bool(const bool value) {
  return value ? T{1} : T{0};
}

template <typename T>
[[nodiscard]] T Expected(const T value, const T lo, const T hi) {
  const T sorted_lo = Min(lo, hi);
  const T sorted_hi = Max(lo, hi);
  const bool inside = value >= lo && value <= hi;
  const bool band_inside = value >= sorted_lo && value <= sorted_hi;
  const T min = Min(Min(value, lo), hi);
  const T max = Max(Max(value, lo), hi);
  const T median = Max(Min(value, lo), Min(Max(value, lo), hi));
  rund::kernel::i128 result = Min(Min(value, lo), hi);
  result += Max(Max(value, lo), hi);
  result += median;
  result += SubSat(max, min);
  result += Clamp(value, sorted_lo, sorted_hi);
  result += band_inside ? value : T{0};
  result += band_inside ? T{0} : value;
  result += inside ? value : hi;
  result += !inside ? lo : value;
  result += Bool<T>(inside);
  result += Bool<T>(!inside);
  const rund::kernel::i128 lower = std::numeric_limits<T>::min();
  const rund::kernel::i128 upper = std::numeric_limits<T>::max();
  return static_cast<T>(result < lower   ? lower
                        : result > upper ? upper
                                         : result);
}

template <typename T>
[[nodiscard]] bool RunRangeOp(std::array<T, 8u> &value, std::array<T, 8u> &lo,
                              std::array<T, 8u> &hi, std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 63>()
          .template read<"value">(value.data())
          .template read<"lo">(lo.data())
          .template read<"hi">(hi.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 31>()
          .template read<"value">(value.data())
          .template read<"lo">(lo.data())
          .template read<"hi">(hi.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-range")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto lo = b.template read<"lo">();
            auto hi = b.template read<"hi">();
            auto out = b.template write<"out">();
            const auto in = rund::compute_dsl::in_range(value[i], lo[i], hi[i]);
            const auto out_of =
                rund::compute_dsl::out_range(value[i], lo[i], hi[i]);
            out[i] = rund::compute_dsl::min(value[i], lo[i], hi[i]) +
                     rund::compute_dsl::max(value[i], lo[i], hi[i]) +
                     rund::compute_dsl::median(value[i], lo[i], hi[i]) +
                     rund::compute_dsl::spread(value[i], lo[i], hi[i]) +
                     rund::compute_dsl::clamp_range(value[i], hi[i], lo[i]) +
                     rund::compute_dsl::bandpass(value[i], hi[i], lo[i]) +
                     rund::compute_dsl::bandstop(value[i], hi[i], lo[i]) +
                     rund::compute_dsl::select(in, value[i], hi[i]) +
                     rund::compute_dsl::select(out_of, lo[i], value[i]) + in +
                     out_of;
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(59u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &value, std::array<T, 8u> &lo,
                           std::array<T, 8u> &hi) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunRangeOp(value, lo, hi, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] == Expected(value[index], lo[index], hi[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicRangeDslOps() {
  std::array<rund::kernel::i32, 8u> value32{-12, -4, 0, 5, 11, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> lo32{-10, -4, 1, 5, 9, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> hi32{10, 6, 7, 5, 15, 30, -3, -8};
  TEST_ASSERT(RunCase(value32, lo32, hi32));

  std::array<rund::kernel::i64, 8u> value64{
      -12, -4, 0, 5, 0x1000000000000000ll, 19, -21, 7};
  std::array<rund::kernel::i64, 8u> lo64{-10, -4, 1, 5, 0x0800000000000000ll,
                                         20,  3,  12};
  std::array<rund::kernel::i64, 8u> hi64{10, 6,  7, 5, 0x1800000000000000ll,
                                         30, -3, -8};
  return RunCase(value64, lo64, hi64);
}

} // namespace node_accel_contract
