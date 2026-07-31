#include "fixed.hpp"
#include "stat.hpp"
#include "wide.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T> [[nodiscard]] constexpr T MaxFixed() noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return 0x7fffffffffffffffll;
  } else {
    return 0x7fffffff;
  }
}

template <typename T> [[nodiscard]] T Sat01(const T value) {
  if (value < T{0}) {
    return T{0};
  }
  return value > MaxFixed<T>() ? MaxFixed<T>() : value;
}

template <typename T> [[nodiscard]] T Lerp(const T a, const T b, const T t) {
  const T tt = Sat01(t);
  return cpu::stat::AddSat(
      cpu::stat::MulFixed(a, cpu::stat::SubSat(MaxFixed<T>(), tt)),
      cpu::stat::MulFixed(b, tt));
}

template <typename T> [[nodiscard]] T Fade(const T t) {
  const T tt = Sat01(t);
  const T t2 = cpu::stat::MulFixed(tt, tt);
  const T inv = cpu::stat::SubSat(MaxFixed<T>(), tt);
  const T bump = cpu::stat::MulFixed(t2, inv);
  return cpu::stat::AddSat(t2, cpu::stat::AddSat(bump, bump));
}

template <typename T>
[[nodiscard]] T Bilerp(const T x00, const T x10, const T x01, const T x11,
                       const T tx, const T ty) {
  return Lerp(Lerp(x00, x10, tx), Lerp(x01, x11, tx), ty);
}

template <typename T>
[[nodiscard]] T Trilerp(const T x000, const T x100, const T x010, const T x110,
                        const T x001, const T x101, const T x011, const T x111,
                        const T tx, const T ty, const T tz) {
  return Lerp(Bilerp(x000, x100, x010, x110, tx, ty),
              Bilerp(x001, x101, x011, x111, tx, ty), tz);
}

template <typename T>
[[nodiscard]] T SmoothBilerp(const T a, const T b, const T c, const T d,
                             const T t, const T u) {
  return Bilerp(a, b, c, d, Fade(t), Fade(u));
}

template <typename T>
[[nodiscard]] T Bezier(const T a, const T b, const T c, const T t) {
  return Lerp(Lerp(a, b, t), Lerp(b, c, t), t);
}

template <typename T>
[[nodiscard]] T Bezier(const T a, const T b, const T c, const T d, const T t) {
  return Lerp(Bezier(a, b, c, t), Bezier(b, c, d, t), t);
}

template <typename T>
[[nodiscard]] T Expected(const T a, const T b, const T c, const T d, const T t,
                         const T u, const T v) {
  return cpu::fixed::QuantizeSum<T>(
      Trilerp(a, b, c, d, b, c, d, a, t, u, v), Lerp(a, b, Fade(t)),
      SmoothBilerp(a, b, c, d, t, u),
      Trilerp(a, b, c, d, b, c, d, a, Fade(t), Fade(u), Fade(v)),
      Bezier(a, b, c, t), Bezier(a, b, c, d, t));
}

template <typename T>
[[nodiscard]] bool RunInterpOp(std::array<T, 8u> &a, std::array<T, 8u> &b,
                               std::array<T, 8u> &c, std::array<T, 8u> &d,
                               std::array<T, 8u> &t, std::array<T, 8u> &u,
                               std::array<T, 8u> &v, std::array<T, 8u> &out) {
  const auto body = [&]() {
    auto builder = rund::compute_dsl::bind(a.size())
                       .template read<"a">(a.data())
                       .template read<"b">(b.data())
                       .template read<"c">(c.data())
                       .template read<"d">(d.data())
                       .template read<"t">(t.data())
                       .template read<"u">(u.data())
                       .template read<"v">(v.data())
                       .template write<"out">(out.data());
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return builder.template fixed<1, 63>();
    } else {
      return builder.template fixed<1, 31>();
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-interp")
          .on(body)
          .map([](auto i, auto bind) {
            auto a = bind.template read<"a">();
            auto b = bind.template read<"b">();
            auto c = bind.template read<"c">();
            auto d = bind.template read<"d">();
            auto t = bind.template read<"t">();
            auto u = bind.template read<"u">();
            auto v = bind.template read<"v">();
            auto out = bind.template write<"out">();
            out[i] =
                rund::compute_dsl::lerp(a[i], b[i], c[i], d[i], b[i], c[i],
                                        d[i], a[i], t[i], u[i], v[i]) +
                rund::compute_dsl::lerp(rund::compute_dsl::LerpOp::Smooth, a[i],
                                        b[i], t[i]) +
                rund::compute_dsl::lerp(rund::compute_dsl::LerpOp::Smooth, a[i],
                                        b[i], c[i], d[i], t[i], u[i]) +
                rund::compute_dsl::lerp(rund::compute_dsl::LerpOp::Smooth, a[i],
                                        b[i], c[i], d[i], b[i], c[i], d[i],
                                        a[i], t[i], u[i], v[i]) +
                rund::compute_dsl::bezier(a[i], b[i], c[i], t[i]) +
                rund::compute_dsl::bezier(a[i], b[i], c[i], d[i], t[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(87u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &a, std::array<T, 8u> &b,
                           std::array<T, 8u> &c, std::array<T, 8u> &d,
                           std::array<T, 8u> &t, std::array<T, 8u> &u,
                           std::array<T, 8u> &v) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunInterpOp(a, b, c, d, t, u, v, out));
  for (std::size_t i = 0u; i < out.size(); ++i) {
    TEST_ASSERT(out[i] == Expected(a[i], b[i], c[i], d[i], t[i], u[i], v[i]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicInterpDslOps() {
  std::array<rund::kernel::i32, 8u> a{0x10000000, -2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i32, 8u> b{0x08000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> c{0x04000000, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i32, 8u> d{0x02000000, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i32, 8u> t{0x20000000, -3, 0x60000000, 4,
                                      0x7fffffff, 8,  -7,         6};
  std::array<rund::kernel::i32, 8u> u{0x10000000, 5, -9, 4, 2, 8, -7, 6};
  std::array<rund::kernel::i32, 8u> v{0x30000000, -7, 11, -13, 17, 19, -23, 29};
  TEST_ASSERT(RunCase(a, b, c, d, t, u, v));

  std::array<rund::kernel::i64, 8u> a64 = cpu::Widen(a);
  std::array<rund::kernel::i64, 8u> b64 = cpu::Widen(b);
  std::array<rund::kernel::i64, 8u> c64 = cpu::Widen(c);
  std::array<rund::kernel::i64, 8u> d64 = cpu::Widen(d);
  std::array<rund::kernel::i64, 8u> t64 = cpu::Widen(t);
  std::array<rund::kernel::i64, 8u> u64 = cpu::Widen(u);
  std::array<rund::kernel::i64, 8u> v64 = cpu::Widen(v);
  a64[0] = 0x1000000000000000ll;
  b64[0] = 0x0800000000000000ll;
  t64[0] = 0x2000000000000000ll;
  return RunCase(a64, b64, c64, d64, t64, u64, v64);
}

} // namespace node_accel_contract
