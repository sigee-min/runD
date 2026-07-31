#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T> [[nodiscard]] T Abs(const T value) noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAbs(value);
  } else {
    return rund::math32::detail::ScalarAbs(value);
  }
}

template <typename T> [[nodiscard]] T SubSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarSubSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarSubSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T Bool(const bool value) {
  return value ? T{1} : T{0};
}

template <typename T>
[[nodiscard]] T Expected(const T value, const T target, const T tol) {
  const bool near = Abs(SubSat(value, target)) <= Abs(tol);
  const bool near_origin = Abs(value) <= Abs(tol);
  rund::kernel::i128 result = Bool<T>(near);
  result += Bool<T>(near_origin);
  result += near_origin ? T{0} : value;
  result += near ? target : value;
  const rund::kernel::i128 lower = std::numeric_limits<T>::min();
  const rund::kernel::i128 upper = std::numeric_limits<T>::max();
  return static_cast<T>(result < lower   ? lower
                        : result > upper ? upper
                                         : result);
}

template <typename T>
[[nodiscard]] bool RunTolOp(std::array<T, 8u> &value, std::array<T, 8u> &target,
                            std::array<T, 8u> &tol, std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 63>()
          .template read<"value">(value.data())
          .template read<"target">(target.data())
          .template read<"tol">(tol.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 31>()
          .template read<"value">(value.data())
          .template read<"target">(target.data())
          .template read<"tol">(tol.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-tol")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto target = b.template read<"target">();
            auto tol = b.template read<"tol">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::near(value[i], target[i], tol[i]) +
                     rund::compute_dsl::near(value[i], tol[i]) +
                     rund::compute_dsl::deadzone(value[i], tol[i]) +
                     rund::compute_dsl::snap(value[i], target[i], tol[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(62u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &value, std::array<T, 8u> &target,
                           std::array<T, 8u> &tol) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunTolOp(value, target, tol, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    if (out[index] != Expected(value[index], target[index], tol[index])) {
      std::cerr << "tol mismatch index=" << index << " value=" << value[index]
                << " target=" << target[index] << " tol=" << tol[index]
                << " actual=" << out[index] << " expected="
                << Expected(value[index], target[index], tol[index]) << '\n';
    }
    TEST_ASSERT(out[index] ==
                Expected(value[index], target[index], tol[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicTolDslOps() {
  std::array<rund::kernel::i32, 8u> value32{-12, -5, 0, 4, 9, 15, -20, 30};
  std::array<rund::kernel::i32, 8u> target32{-10, -3, 2, 8, 6, 21, -12, 22};
  std::array<rund::kernel::i32, 8u> tol32{2, -1, 3, 0, -4, 5, 7, -8};
  TEST_ASSERT(RunCase(value32, target32, tol32));

  std::array<rund::kernel::i64, 8u> value64{
      -12, -5, 0, 4, 0x1000000000000000ll, 15, -20, 30};
  std::array<rund::kernel::i64, 8u> target64{
      -10, -3, 2, 8, 0x1000000000000003ll, 21, -12, 22};
  std::array<rund::kernel::i64, 8u> tol64{2, -1, 3, 0, -4, 5, 7, -8};
  return RunCase(value64, target64, tol64);
}

} // namespace node_accel_contract
