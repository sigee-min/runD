#include "fixed.hpp"
#include "stat.hpp"
#include "wide.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Poly(const T x, const T c0, const T c1, const T c2) {
  return cpu::stat::AddSat(
      c0, cpu::stat::MulFixed(
              x, cpu::stat::AddSat(c1, cpu::stat::MulFixed(c2, x))));
}

template <typename T>
[[nodiscard]] T Poly(const T x, const T c0, const T c1, const T c2,
                     const T c3) {
  return cpu::stat::AddSat(
      c0,
      cpu::stat::MulFixed(
          x,
          cpu::stat::AddSat(
              c1, cpu::stat::MulFixed(
                      x, cpu::stat::AddSat(c2, cpu::stat::MulFixed(c3, x))))));
}

template <typename T>
[[nodiscard]] bool RunPolyOp(std::array<T, 8u> &x, std::array<T, 8u> &c0,
                             std::array<T, 8u> &c1, std::array<T, 8u> &c2,
                             std::array<T, 8u> &c3, std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 63>()
          .template read<"x">(x.data())
          .template read<"c0">(c0.data())
          .template read<"c1">(c1.data())
          .template read<"c2">(c2.data())
          .template read<"c3">(c3.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 31>()
          .template read<"x">(x.data())
          .template read<"c0">(c0.data())
          .template read<"c1">(c1.data())
          .template read<"c2">(c2.data())
          .template read<"c3">(c3.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-poly")
          .on(body)
          .map([](auto i, auto bind) {
            auto x = bind.template read<"x">();
            auto c0 = bind.template read<"c0">();
            auto c1 = bind.template read<"c1">();
            auto c2 = bind.template read<"c2">();
            auto c3 = bind.template read<"c3">();
            auto out = bind.template write<"out">();
            out[i] = rund::compute_dsl::poly(x[i], c0[i], c1[i], c2[i]) +
                     rund::compute_dsl::poly(x[i], c0[i], c1[i], c2[i], c3[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(86u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &x, std::array<T, 8u> &c0,
                           std::array<T, 8u> &c1, std::array<T, 8u> &c2,
                           std::array<T, 8u> &c3) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunPolyOp(x, c0, c1, c2, c3, out));
  for (std::size_t i = 0u; i < out.size(); ++i) {
    TEST_ASSERT(out[i] == cpu::fixed::QuantizeSum<T>(
                              Poly(x[i], c0[i], c1[i], c2[i]),
                              Poly(x[i], c0[i], c1[i], c2[i], c3[i])));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicPolyDslOps() {
  std::array<rund::kernel::i32, 8u> x{0x10000000, -2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i32, 8u> c0{0x08000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> c1{0x04000000, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i32, 8u> c2{0x02000000, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i32, 8u> c3{0x01000000, 9, -7, 2, 6, -1, 8, 4};
  TEST_ASSERT(RunCase(x, c0, c1, c2, c3));

  std::array<rund::kernel::i64, 8u> x64 = cpu::Widen(x);
  std::array<rund::kernel::i64, 8u> c064 = cpu::Widen(c0);
  std::array<rund::kernel::i64, 8u> c164 = cpu::Widen(c1);
  std::array<rund::kernel::i64, 8u> c264 = cpu::Widen(c2);
  std::array<rund::kernel::i64, 8u> c364 = cpu::Widen(c3);
  x64[0] = 0x1000000000000000ll;
  c064[0] = 0x0800000000000000ll;
  c164[0] = 0x0400000000000000ll;
  return RunCase(x64, c064, c164, c264, c364);
}

} // namespace node_accel_contract
