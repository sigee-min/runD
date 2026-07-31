#include "fixed.hpp"
#include "stat.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Dot(const T a0, const T a1, const T a2, const T b0, const T b1,
                    const T b2) {
  return cpu::stat::AddSat(cpu::stat::AddSat(cpu::stat::MulFixed(a0, b0),
                                             cpu::stat::MulFixed(a1, b1)),
                           cpu::stat::MulFixed(a2, b2));
}

template <typename T>
[[nodiscard]] T Dot(const T a0, const T a1, const T a2, const T a3, const T b0,
                    const T b1, const T b2, const T b3) {
  return cpu::stat::AddSat(cpu::stat::AddSat(cpu::stat::MulFixed(a0, b0),
                                             cpu::stat::MulFixed(a1, b1)),
                           cpu::stat::AddSat(cpu::stat::MulFixed(a2, b2),
                                             cpu::stat::MulFixed(a3, b3)));
}

template <typename T>
[[nodiscard]] T Dot(const T a0, const T a1, const T a2, const T a3, const T a4,
                    const T b0, const T b1, const T b2, const T b3,
                    const T b4) {
  return cpu::stat::AddSat(Dot(a0, a1, a2, a3, b0, b1, b2, b3),
                           cpu::stat::MulFixed(a4, b4));
}

template <typename T>
[[nodiscard]] T Dot(const T a0, const T a1, const T a2, const T a3, const T a4,
                    const T a5, const T b0, const T b1, const T b2, const T b3,
                    const T b4, const T b5) {
  return cpu::stat::AddSat(Dot(a0, a1, a2, b0, b1, b2),
                           Dot(a3, a4, a5, b3, b4, b5));
}

template <typename T>
[[nodiscard]] T Dot(const T a0, const T a1, const T a2, const T a3, const T a4,
                    const T a5, const T a6, const T a7, const T b0, const T b1,
                    const T b2, const T b3, const T b4, const T b5, const T b6,
                    const T b7) {
  return cpu::stat::AddSat(Dot(a0, a1, a2, a3, b0, b1, b2, b3),
                           Dot(a4, a5, a6, a7, b4, b5, b6, b7));
}

template <typename T>
[[nodiscard]] T Expected(const T x0, const T x1, const T x2, const T x3,
                         const T x4, const T x5, const T x6, const T x7,
                         const T x8, const T k0, const T k1, const T k2,
                         const T k3, const T k4, const T k5, const T k6,
                         const T k7, const T k8) {
  return cpu::fixed::QuantizeSum<T>(
      Dot(x0, x1, x2, x3, k0, k1, k2, k3),
      Dot(x0, x1, x2, x3, x4, k0, k1, k2, k3, k4),
      Dot(x0, x1, x2, x3, x4, x5, k0, k1, k2, k3, k4, k5),
      Dot(x0, x1, x2, x3, x4, x5, x6, x7, k0, k1, k2, k3, k4, k5, k6, k7),
      Dot(x0, x1, x2, k0, k1, k2), Dot(x0, x1, x2, x3, x4, k0, k1, k2, k3, k4),
      cpu::stat::AddSat(Dot(x0, x1, x2, x3, x4, x5, k0, k1, k2, k3, k4, k5),
                        cpu::stat::MulFixed(x6, k6)),
      cpu::stat::AddSat(
          Dot(x0, x1, x2, x3, x4, x5, x6, x7, k0, k1, k2, k3, k4, k5, k6, k7),
          cpu::stat::MulFixed(x8, k8)));
}

template <typename T>
[[nodiscard]] bool
RunLinearOp(std::array<T, 8u> &x0, std::array<T, 8u> &x1, std::array<T, 8u> &x2,
            std::array<T, 8u> &x3, std::array<T, 8u> &x4, std::array<T, 8u> &x5,
            std::array<T, 8u> &x6, std::array<T, 8u> &x7, std::array<T, 8u> &x8,
            std::array<T, 8u> &k0, std::array<T, 8u> &k1, std::array<T, 8u> &k2,
            std::array<T, 8u> &k3, std::array<T, 8u> &k4, std::array<T, 8u> &k5,
            std::array<T, 8u> &k6, std::array<T, 8u> &k7, std::array<T, 8u> &k8,
            std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(x0.size())
          .template fixed<1, 63>()
          .template read<"x0">(x0.data())
          .template read<"x1">(x1.data())
          .template read<"x2">(x2.data())
          .template read<"x3">(x3.data())
          .template read<"x4">(x4.data())
          .template read<"x5">(x5.data())
          .template read<"x6">(x6.data())
          .template read<"x7">(x7.data())
          .template read<"x8">(x8.data())
          .template read<"k0">(k0.data())
          .template read<"k1">(k1.data())
          .template read<"k2">(k2.data())
          .template read<"k3">(k3.data())
          .template read<"k4">(k4.data())
          .template read<"k5">(k5.data())
          .template read<"k6">(k6.data())
          .template read<"k7">(k7.data())
          .template read<"k8">(k8.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(x0.size())
          .template fixed<1, 31>()
          .template read<"x0">(x0.data())
          .template read<"x1">(x1.data())
          .template read<"x2">(x2.data())
          .template read<"x3">(x3.data())
          .template read<"x4">(x4.data())
          .template read<"x5">(x5.data())
          .template read<"x6">(x6.data())
          .template read<"x7">(x7.data())
          .template read<"x8">(x8.data())
          .template read<"k0">(k0.data())
          .template read<"k1">(k1.data())
          .template read<"k2">(k2.data())
          .template read<"k3">(k3.data())
          .template read<"k4">(k4.data())
          .template read<"k5">(k5.data())
          .template read<"k6">(k6.data())
          .template read<"k7">(k7.data())
          .template read<"k8">(k8.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-linear")
          .on(body)
          .map([](auto i, auto bind) {
            auto x0 = bind.template read<"x0">();
            auto x1 = bind.template read<"x1">();
            auto x2 = bind.template read<"x2">();
            auto x3 = bind.template read<"x3">();
            auto x4 = bind.template read<"x4">();
            auto x5 = bind.template read<"x5">();
            auto x6 = bind.template read<"x6">();
            auto x7 = bind.template read<"x7">();
            auto x8 = bind.template read<"x8">();
            auto k0 = bind.template read<"k0">();
            auto k1 = bind.template read<"k1">();
            auto k2 = bind.template read<"k2">();
            auto k3 = bind.template read<"k3">();
            auto k4 = bind.template read<"k4">();
            auto k5 = bind.template read<"k5">();
            auto k6 = bind.template read<"k6">();
            auto k7 = bind.template read<"k7">();
            auto k8 = bind.template read<"k8">();
            auto out = bind.template write<"out">();
            out[i] =
                rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], k0[i], k1[i],
                                       k2[i], k3[i]) +
                rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], k0[i],
                                       k1[i], k2[i], k3[i], k4[i]) +
                rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                                       k0[i], k1[i], k2[i], k3[i], k4[i],
                                       k5[i]) +
                rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                                       x6[i], x7[i], k0[i], k1[i], k2[i], k3[i],
                                       k4[i], k5[i], k6[i], k7[i]) +
                rund::compute_dsl::conv(x0[i], x1[i], x2[i], k0[i], k1[i],
                                        k2[i]) +
                rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i],
                                        k0[i], k1[i], k2[i], k3[i], k4[i]) +
                rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i],
                                        x5[i], x6[i], k0[i], k1[i], k2[i],
                                        k3[i], k4[i], k5[i], k6[i]) +
                rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i],
                                        x5[i], x6[i], x7[i], x8[i], k0[i],
                                        k1[i], k2[i], k3[i], k4[i], k5[i],
                                        k6[i], k7[i], k8[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(71u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool
RunCase(std::array<T, 8u> &x0, std::array<T, 8u> &x1, std::array<T, 8u> &x2,
        std::array<T, 8u> &x3, std::array<T, 8u> &x4, std::array<T, 8u> &x5,
        std::array<T, 8u> &x6, std::array<T, 8u> &x7, std::array<T, 8u> &x8,
        std::array<T, 8u> &k0, std::array<T, 8u> &k1, std::array<T, 8u> &k2,
        std::array<T, 8u> &k3, std::array<T, 8u> &k4, std::array<T, 8u> &k5,
        std::array<T, 8u> &k6, std::array<T, 8u> &k7, std::array<T, 8u> &k8) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunLinearOp(x0, x1, x2, x3, x4, x5, x6, x7, x8, k0, k1, k2, k3,
                          k4, k5, k6, k7, k8, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] ==
                Expected(x0[index], x1[index], x2[index], x3[index], x4[index],
                         x5[index], x6[index], x7[index], x8[index], k0[index],
                         k1[index], k2[index], k3[index], k4[index], k5[index],
                         k6[index], k7[index], k8[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicLinearDslOps() {
  std::array<rund::kernel::i32, 8u> x0{-12, -4, 0, 5, 11, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> x1{-10, -1, 1, 3, 9, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> x2{10, 6, 7, 5, 15, 30, -3, -8};
  std::array<rund::kernel::i32, 8u> x3{3, 4, -2, 9, 8, -11, 2, 1};
  std::array<rund::kernel::i32, 8u> x4{7, -5, 0, 13, 6, -2, 4, 5};
  std::array<rund::kernel::i32, 8u> x5{2, 8, -6, 4, 10, -7, 1, 3};
  std::array<rund::kernel::i32, 8u> x6{-3, 2, 9, -1, 5, 12, -4, 6};
  std::array<rund::kernel::i32, 8u> x7{5, -9, 11, 2, -8, 1, 7, -6};
  std::array<rund::kernel::i32, 8u> x8{4, 3, -5, 8, 2, -10, 6, 9};
  std::array<rund::kernel::i32, 8u> k0{0x10000000, 2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i32, 8u> k1{0x08000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> k2{0x04000000, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i32, 8u> k3{0x02000000, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i32, 8u> k4{0x01000000, 9, -7, 2, 6, -1, 8, 4};
  std::array<rund::kernel::i32, 8u> k5{0x00800000, -3, 6, 1, -5, 7, 2, 10};
  std::array<rund::kernel::i32, 8u> k6{0x00400000, 4, -8, 3, 9, -2, 5, -1};
  std::array<rund::kernel::i32, 8u> k7{0x00200000, -6, 1, 7, -4, 3, 11, 2};
  std::array<rund::kernel::i32, 8u> k8{0x00100000, 5, -9, 4, 2, 8, -7, 6};
  TEST_ASSERT(RunCase(x0, x1, x2, x3, x4, x5, x6, x7, x8, k0, k1, k2, k3, k4,
                      k5, k6, k7, k8));

  std::array<rund::kernel::i64, 8u> x064{-12, -4,  0, 5, 0x1000000000000000ll,
                                         19,  -21, 7};
  std::array<rund::kernel::i64, 8u> x164{-10, -1, 1, 3, 0x0800000000000000ll,
                                         20,  3,  12};
  std::array<rund::kernel::i64, 8u> x264{10, 6,  7, 5, 0x1800000000000000ll,
                                         30, -3, -8};
  std::array<rund::kernel::i64, 8u> x364{3,   4, -2, 9, 0x0400000000000000ll,
                                         -11, 2, 1};
  std::array<rund::kernel::i64, 8u> x464{7,  -5, 0, 13, 0x0c00000000000000ll,
                                         -2, 4,  5};
  std::array<rund::kernel::i64, 8u> x564{2,  8, -6, 4, 0x0600000000000000ll,
                                         -7, 1, 3};
  std::array<rund::kernel::i64, 8u> x664{-3, 2,  9, -1, 0x0200000000000000ll,
                                         12, -4, 6};
  std::array<rund::kernel::i64, 8u> x764{5, -9, 11, 2, 0x0100000000000000ll,
                                         1, 7,  -6};
  std::array<rund::kernel::i64, 8u> x864{4,   3, -5, 8, 0x0080000000000000ll,
                                         -10, 6, 9};
  std::array<rund::kernel::i64, 8u> k064{
      0x1000000000000000ll, 2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i64, 8u> k164{
      0x0800000000000000ll, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i64, 8u> k264{
      0x0400000000000000ll, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i64, 8u> k364{
      0x0200000000000000ll, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i64, 8u> k464{
      0x0100000000000000ll, 9, -7, 2, 6, -1, 8, 4};
  std::array<rund::kernel::i64, 8u> k564{
      0x0080000000000000ll, -3, 6, 1, -5, 7, 2, 10};
  std::array<rund::kernel::i64, 8u> k664{
      0x0040000000000000ll, 4, -8, 3, 9, -2, 5, -1};
  std::array<rund::kernel::i64, 8u> k764{
      0x0020000000000000ll, -6, 1, 7, -4, 3, 11, 2};
  std::array<rund::kernel::i64, 8u> k864{
      0x0010000000000000ll, 5, -9, 4, 2, 8, -7, 6};
  return RunCase(x064, x164, x264, x364, x464, x564, x664, x764, x864, k064,
                 k164, k264, k364, k464, k564, k664, k764, k864);
}

} // namespace node_accel_contract
