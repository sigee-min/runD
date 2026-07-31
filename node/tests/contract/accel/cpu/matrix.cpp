#include "fixed.hpp"
#include "stat.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Dot(const T ax, const T ay, const T bx, const T by) {
  return cpu::stat::AddSat(cpu::stat::MulFixed(ax, bx),
                           cpu::stat::MulFixed(ay, by));
}

template <typename T>
[[nodiscard]] T Dot(const T ax, const T ay, const T az, const T bx, const T by,
                    const T bz) {
  return cpu::stat::AddSat(Dot(ax, ay, bx, by), cpu::stat::MulFixed(az, bz));
}

template <typename T>
[[nodiscard]] T MatDet(const T m00, const T m01, const T m10, const T m11) {
  return cpu::stat::SubSat(cpu::stat::MulFixed(m00, m11),
                           cpu::stat::MulFixed(m01, m10));
}

template <typename T>
[[nodiscard]] T MatDet(const T m00, const T m01, const T m02, const T m10,
                       const T m11, const T m12, const T m20, const T m21,
                       const T m22) {
  const T a = cpu::stat::MulFixed(m00, MatDet(m11, m12, m21, m22));
  const T b = cpu::stat::MulFixed(m01, MatDet(m10, m12, m20, m22));
  const T c = cpu::stat::MulFixed(m02, MatDet(m10, m11, m20, m21));
  return cpu::stat::AddSat(cpu::stat::SubSat(a, b), c);
}

template <typename T>
[[nodiscard]] T Triple(const T ax, const T ay, const T az, const T bx,
                       const T by, const T bz, const T cx, const T cy,
                       const T cz) {
  return Dot(ax, ay, az, MatDet(by, bz, cy, cz), MatDet(bz, bx, cz, cx),
             MatDet(bx, by, cx, cy));
}

template <typename T>
[[nodiscard]] T Expected(const T m00, const T m01, const T m02, const T m10,
                         const T m11, const T m12, const T m20, const T m21,
                         const T m22, const T x, const T y, const T z) {
  return cpu::fixed::QuantizeSum<T>(
      Dot(m00, m01, x, y), Dot(m10, m11, x, y), MatDet(m00, m01, m10, m11),
      MatDet(m00, m01, m02, m10, m11, m12, m20, m21, m22),
      Triple(m00, m01, m02, m10, m11, m12, m20, m21, m22),
      Dot(m00, m01, m02, x, y, z), Dot(m10, m11, m12, x, y, z),
      Dot(m20, m21, m22, x, y, z));
}

template <typename T>
[[nodiscard]] bool RunMatrixOp(std::array<T, 8u> &m00, std::array<T, 8u> &m01,
                               std::array<T, 8u> &m02, std::array<T, 8u> &m10,
                               std::array<T, 8u> &m11, std::array<T, 8u> &m12,
                               std::array<T, 8u> &m20, std::array<T, 8u> &m21,
                               std::array<T, 8u> &m22, std::array<T, 8u> &x,
                               std::array<T, 8u> &y, std::array<T, 8u> &z,
                               std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 63>()
          .template read<"m00">(m00.data())
          .template read<"m01">(m01.data())
          .template read<"m02">(m02.data())
          .template read<"m10">(m10.data())
          .template read<"m11">(m11.data())
          .template read<"m12">(m12.data())
          .template read<"m20">(m20.data())
          .template read<"m21">(m21.data())
          .template read<"m22">(m22.data())
          .template read<"x">(x.data())
          .template read<"y">(y.data())
          .template read<"z">(z.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 31>()
          .template read<"m00">(m00.data())
          .template read<"m01">(m01.data())
          .template read<"m02">(m02.data())
          .template read<"m10">(m10.data())
          .template read<"m11">(m11.data())
          .template read<"m12">(m12.data())
          .template read<"m20">(m20.data())
          .template read<"m21">(m21.data())
          .template read<"m22">(m22.data())
          .template read<"x">(x.data())
          .template read<"y">(y.data())
          .template read<"z">(z.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-matrix")
          .on(body)
          .map([](auto i, auto bind) {
            auto m00 = bind.template read<"m00">();
            auto m01 = bind.template read<"m01">();
            auto m02 = bind.template read<"m02">();
            auto m10 = bind.template read<"m10">();
            auto m11 = bind.template read<"m11">();
            auto m12 = bind.template read<"m12">();
            auto m20 = bind.template read<"m20">();
            auto m21 = bind.template read<"m21">();
            auto m22 = bind.template read<"m22">();
            auto x = bind.template read<"x">();
            auto y = bind.template read<"y">();
            auto z = bind.template read<"z">();
            auto out = bind.template write<"out">();
            out[i] =
                rund::compute_dsl::mat(rund::compute_dsl::Axis::X, m00[i],
                                       m01[i], x[i], y[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::Axis::Y, m10[i],
                                       m11[i], x[i], y[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::MatOp::Determinant,
                                       m00[i], m01[i], m10[i], m11[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::MatOp::Determinant,
                                       m00[i], m01[i], m02[i], m10[i], m11[i],
                                       m12[i], m20[i], m21[i], m22[i]) +
                rund::compute_dsl::triple(m00[i], m01[i], m02[i], m10[i],
                                          m11[i], m12[i], m20[i], m21[i],
                                          m22[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::Axis::X, m00[i],
                                       m01[i], m02[i], x[i], y[i], z[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::Axis::Y, m10[i],
                                       m11[i], m12[i], x[i], y[i], z[i]) +
                rund::compute_dsl::mat(rund::compute_dsl::Axis::Z, m20[i],
                                       m21[i], m22[i], x[i], y[i], z[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(83u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool
RunCase(std::array<T, 8u> &m00, std::array<T, 8u> &m01, std::array<T, 8u> &m02,
        std::array<T, 8u> &m10, std::array<T, 8u> &m11, std::array<T, 8u> &m12,
        std::array<T, 8u> &m20, std::array<T, 8u> &m21, std::array<T, 8u> &m22,
        std::array<T, 8u> &x, std::array<T, 8u> &y, std::array<T, 8u> &z) {
  std::array<T, 8u> out{};
  TEST_ASSERT(
      RunMatrixOp(m00, m01, m02, m10, m11, m12, m20, m21, m22, x, y, z, out));
  for (std::size_t i = 0u; i < out.size(); ++i) {
    TEST_ASSERT(out[i] == Expected(m00[i], m01[i], m02[i], m10[i], m11[i],
                                   m12[i], m20[i], m21[i], m22[i], x[i], y[i],
                                   z[i]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicMatrixDslOps() {
  std::array<rund::kernel::i32, 8u> m00{0x10000000, -2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i32, 8u> m01{0x08000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> m02{0x04000000, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i32, 8u> m10{0x02000000, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i32, 8u> m11{0x01000000, 9, -7, 2, 6, -1, 8, 4};
  std::array<rund::kernel::i32, 8u> m12{0x00800000, -3, 6, 1, -5, 7, 2, 10};
  std::array<rund::kernel::i32, 8u> m20{0x00400000, 4, -8, 3, 9, -2, 5, -1};
  std::array<rund::kernel::i32, 8u> m21{0x00200000, -6, 1, 7, -4, 3, 11, 2};
  std::array<rund::kernel::i32, 8u> m22{0x00100000, 5, -9, 4, 2, 8, -7, 6};
  std::array<rund::kernel::i32, 8u> x32{-12, -4, 0, 5, 11, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> y32{-10, -1, 1, 3, 9, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> z32{10, 6, 7, 5, 15, 30, -3, -8};
  TEST_ASSERT(
      RunCase(m00, m01, m02, m10, m11, m12, m20, m21, m22, x32, y32, z32));

  std::array<rund::kernel::i64, 8u> m0064{
      0x1000000000000000ll, -2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i64, 8u> m0164{
      0x0800000000000000ll, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i64, 8u> m0264{
      0x0400000000000000ll, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i64, 8u> m1064{
      0x0200000000000000ll, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i64, 8u> m1164{
      0x0100000000000000ll, 9, -7, 2, 6, -1, 8, 4};
  std::array<rund::kernel::i64, 8u> m1264{
      0x0080000000000000ll, -3, 6, 1, -5, 7, 2, 10};
  std::array<rund::kernel::i64, 8u> m2064{
      0x0040000000000000ll, 4, -8, 3, 9, -2, 5, -1};
  std::array<rund::kernel::i64, 8u> m2164{
      0x0020000000000000ll, -6, 1, 7, -4, 3, 11, 2};
  std::array<rund::kernel::i64, 8u> m2264{
      0x0010000000000000ll, 5, -9, 4, 2, 8, -7, 6};
  std::array<rund::kernel::i64, 8u> x64{-12, -4,  0, 5, 0x1000000000000000ll,
                                        19,  -21, 7};
  std::array<rund::kernel::i64, 8u> y64{-10, -1, 1, 3, 0x0800000000000000ll,
                                        20,  3,  12};
  std::array<rund::kernel::i64, 8u> z64{10, 6,  7, 5, 0x1800000000000000ll,
                                        30, -3, -8};
  return RunCase(m0064, m0164, m0264, m1064, m1164, m1264, m2064, m2164, m2264,
                 x64, y64, z64);
}

} // namespace node_accel_contract
