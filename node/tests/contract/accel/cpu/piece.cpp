#include "fixed.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T> [[nodiscard]] T Abs(const T value) noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAbs(value);
  } else {
    return rund::math32::detail::ScalarAbs(value);
  }
}

template <typename T>
[[nodiscard]] T Clamp(const T value, const T lo, const T hi) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarClamp(value, lo, hi);
  } else {
    return rund::math32::detail::ScalarClamp(value, lo, hi);
  }
}

template <typename T> [[nodiscard]] T NegPositiveFixed(const T value) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarNegPositiveFixed(value);
  } else {
    return rund::math32::detail::ScalarNegPositiveFixed(value);
  }
}

template <typename T> [[nodiscard]] T Expected(const T value, const T limit) {
  const T bound = Abs(limit);
  return cpu::fixed::QuantizeSum<T>(
      Clamp(value, NegPositiveFixed(bound), bound), value > T{0} ? value : T{0},
      value < T{0} ? value : T{0});
}

template <typename T>
[[nodiscard]] bool RunPieceOp(std::array<T, 8u> &value,
                              std::array<T, 8u> &limit,
                              std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 63>()
          .template read<"value">(value.data())
          .template read<"limit">(limit.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 31>()
          .template read<"value">(value.data())
          .template read<"limit">(limit.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-piece")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto limit = b.template read<"limit">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::clip(value[i], limit[i]) +
                     rund::compute_dsl::positive_part(value[i]) +
                     rund::compute_dsl::negative_part(value[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(63u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &value, std::array<T, 8u> &limit) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunPieceOp(value, limit, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] == Expected(value[index], limit[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicPieceDslOps() {
  std::array<rund::kernel::i32, 8u> value32{-12, -5, 0, 4, 9, 15, -20, 30};
  std::array<rund::kernel::i32, 8u> limit32{8, -3, 0, 6, -12, 10, 7, -25};
  TEST_ASSERT(RunCase(value32, limit32));

  std::array<rund::kernel::i64, 8u> value64{
      -12, -5, 0, 4, 0x1000000000000000ll, 15, -20, 30};
  std::array<rund::kernel::i64, 8u> limit64{8,  -3, 0,  6, 0x0800000000000000ll,
                                            10, 7,  -25};
  return RunCase(value64, limit64);
}

} // namespace node_accel_contract
