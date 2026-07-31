#include "fixed.hpp"
#include "stat.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>
#include <cstdio>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Expected(const T x0, const T x1, const T x2, const T x3,
                         const T y0, const T y1, const T y2, const T y3) {
  return cpu::fixed::QuantizeSum<T>(
      cpu::stat::Cov(x0, x1, y0, y1), cpu::stat::Cov(x0, x1, x2, y0, y1, y2),
      cpu::stat::Cov(x0, x1, x2, x3, y0, y1, y2, y3),
      cpu::stat::Corr(x0, x1, y0, y1), cpu::stat::Corr(x0, x1, x2, y0, y1, y2),
      cpu::stat::Corr(x0, x1, x2, x3, y0, y1, y2, y3));
}

template <typename T>
[[nodiscard]] bool RunCorrOp(std::array<T, 8u> &x0, std::array<T, 8u> &x1,
                             std::array<T, 8u> &x2, std::array<T, 8u> &x3,
                             std::array<T, 8u> &y0, std::array<T, 8u> &y1,
                             std::array<T, 8u> &y2, std::array<T, 8u> &y3,
                             std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(x0.size())
          .template fixed<1, 63>()
          .template read<"x0">(x0.data())
          .template read<"x1">(x1.data())
          .template read<"x2">(x2.data())
          .template read<"x3">(x3.data())
          .template read<"y0">(y0.data())
          .template read<"y1">(y1.data())
          .template read<"y2">(y2.data())
          .template read<"y3">(y3.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(x0.size())
          .template fixed<1, 31>()
          .template read<"x0">(x0.data())
          .template read<"x1">(x1.data())
          .template read<"x2">(x2.data())
          .template read<"x3">(x3.data())
          .template read<"y0">(y0.data())
          .template read<"y1">(y1.data())
          .template read<"y2">(y2.data())
          .template read<"y3">(y3.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-corr")
          .on(body)
          .map([](auto i, auto bind) {
            auto x0 = bind.template read<"x0">();
            auto x1 = bind.template read<"x1">();
            auto x2 = bind.template read<"x2">();
            auto x3 = bind.template read<"x3">();
            auto y0 = bind.template read<"y0">();
            auto y1 = bind.template read<"y1">();
            auto y2 = bind.template read<"y2">();
            auto y3 = bind.template read<"y3">();
            auto out = bind.template write<"out">();
            out[i] = rund::compute_dsl::cov(x0[i], x1[i], y0[i], y1[i]) +
                     rund::compute_dsl::cov(x0[i], x1[i], x2[i], y0[i], y1[i],
                                            y2[i]) +
                     rund::compute_dsl::cov(x0[i], x1[i], x2[i], x3[i], y0[i],
                                            y1[i], y2[i], y3[i]) +
                     rund::compute_dsl::corr(x0[i], x1[i], y0[i], y1[i]) +
                     rund::compute_dsl::corr(x0[i], x1[i], x2[i], y0[i], y1[i],
                                             y2[i]) +
                     rund::compute_dsl::corr(x0[i], x1[i], x2[i], x3[i], y0[i],
                                             y1[i], y2[i], y3[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(67u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &x0, std::array<T, 8u> &x1,
                           std::array<T, 8u> &x2, std::array<T, 8u> &x3,
                           std::array<T, 8u> &y0, std::array<T, 8u> &y1,
                           std::array<T, 8u> &y2, std::array<T, 8u> &y3) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunCorrOp(x0, x1, x2, x3, y0, y1, y2, y3, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    const T expected = Expected(x0[index], x1[index], x2[index], x3[index],
                                y0[index], y1[index], y2[index], y3[index]);
    if (out[index] != expected) {
      std::fprintf(
          stderr,
          "corr mismatch width=%zu index=%zu actual=%lld expected=%lld\n",
          sizeof(T) * 8u, index, static_cast<long long>(out[index]),
          static_cast<long long>(expected));
      return false;
    }
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicCorrDslOps() {
  std::array<rund::kernel::i32, 8u> x0{-12, -4, 0, 5, 0x10000000, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> x1{-10, -1, 1, 3, 0x20000000, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> x2{10, 6, 7, 5, 0x30000000, 30, -3, -8};
  std::array<rund::kernel::i32, 8u> x3{2, -7, 9, 1, 0x08000000, -6, 11, 4};
  std::array<rund::kernel::i32, 8u> y0{3, 4, -2, 9, 0x08000000, -11, 2, 1};
  std::array<rund::kernel::i32, 8u> y1{7, -5, 0, 13, 0x10000000, -2, 4, 5};
  std::array<rund::kernel::i32, 8u> y2{-1, 8, 3, 2, 0x18000000, 10, 6, 9};
  std::array<rund::kernel::i32, 8u> y3{5, -9, 2, 6, 0x20000000, 7, -4, 3};
  TEST_ASSERT(RunCase(x0, x1, x2, x3, y0, y1, y2, y3));

  std::array<rund::kernel::i64, 8u> x064{-12, -4,  0, 5, 0x1000000000000000ll,
                                         19,  -21, 7};
  std::array<rund::kernel::i64, 8u> x164{-10, -1, 1, 3, 0x0800000000000000ll,
                                         20,  3,  12};
  std::array<rund::kernel::i64, 8u> x264{10, 6,  7, 5, 0x1800000000000000ll,
                                         30, -3, -8};
  std::array<rund::kernel::i64, 8u> x364{2,  -7, 9, 1, 0x0400000000000000ll,
                                         -6, 11, 4};
  std::array<rund::kernel::i64, 8u> y064{3,   4, -2, 9, 0x0400000000000000ll,
                                         -11, 2, 1};
  std::array<rund::kernel::i64, 8u> y164{7,  -5, 0, 13, 0x0c00000000000000ll,
                                         -2, 4,  5};
  std::array<rund::kernel::i64, 8u> y264{-1, 8, 3, 2, 0x1400000000000000ll,
                                         10, 6, 9};
  std::array<rund::kernel::i64, 8u> y364{5, -9, 2, 6, 0x1c00000000000000ll,
                                         7, -4, 3};
  return RunCase(x064, x164, x264, x364, y064, y164, y264, y364);
}

} // namespace node_accel_contract
