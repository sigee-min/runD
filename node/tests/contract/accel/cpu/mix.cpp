#include "fixed.hpp"
#include "stat.hpp"
#include "wide.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Mix(const T a, const T b, const T wa, const T wb) {
  return cpu::stat::AddSat(cpu::stat::MulFixed(a, wa),
                           cpu::stat::MulFixed(b, wb));
}

template <typename T>
[[nodiscard]] T Mix(const T a, const T b, const T c, const T wa, const T wb,
                    const T wc) {
  return cpu::stat::AddSat(Mix(a, b, wa, wb), cpu::stat::MulFixed(c, wc));
}

template <typename T>
[[nodiscard]] T Mix(const T a, const T b, const T c, const T d, const T wa,
                    const T wb, const T wc, const T wd) {
  return cpu::stat::AddSat(Mix(a, b, wa, wb), Mix(c, d, wc, wd));
}

template <typename T>
[[nodiscard]] T Expected(const T a, const T b, const T c, const T d, const T wa,
                         const T wb, const T wc, const T wd) {
  return cpu::fixed::QuantizeSum<T>(Mix(a, b, wa, wb), Mix(a, b, c, wa, wb, wc),
                                    Mix(a, b, c, d, wa, wb, wc, wd));
}

template <typename T>
[[nodiscard]] bool
RunMixOp(std::array<T, 8u> &a, std::array<T, 8u> &b, std::array<T, 8u> &c,
         std::array<T, 8u> &d, std::array<T, 8u> &wa, std::array<T, 8u> &wb,
         std::array<T, 8u> &wc, std::array<T, 8u> &wd, std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(a.size())
          .template fixed<1, 63>()
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template read<"d">(d.data())
          .template read<"wa">(wa.data())
          .template read<"wb">(wb.data())
          .template read<"wc">(wc.data())
          .template read<"wd">(wd.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(a.size())
          .template fixed<1, 31>()
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template read<"d">(d.data())
          .template read<"wa">(wa.data())
          .template read<"wb">(wb.data())
          .template read<"wc">(wc.data())
          .template read<"wd">(wd.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-mix")
          .on(body)
          .map([](auto i, auto bind) {
            auto a = bind.template read<"a">();
            auto b = bind.template read<"b">();
            auto c = bind.template read<"c">();
            auto d = bind.template read<"d">();
            auto wa = bind.template read<"wa">();
            auto wb = bind.template read<"wb">();
            auto wc = bind.template read<"wc">();
            auto wd = bind.template read<"wd">();
            auto out = bind.template write<"out">();
            out[i] =
                rund::compute_dsl::mix(a[i], b[i], wa[i], wb[i]) +
                rund::compute_dsl::mix(a[i], b[i], c[i], wa[i], wb[i], wc[i]) +
                rund::compute_dsl::mix(a[i], b[i], c[i], d[i], wa[i], wb[i],
                                       wc[i], wd[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(85u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &a, std::array<T, 8u> &b,
                           std::array<T, 8u> &c, std::array<T, 8u> &d,
                           std::array<T, 8u> &wa, std::array<T, 8u> &wb,
                           std::array<T, 8u> &wc, std::array<T, 8u> &wd) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunMixOp(a, b, c, d, wa, wb, wc, wd, out));
  for (std::size_t i = 0u; i < out.size(); ++i) {
    TEST_ASSERT(out[i] ==
                Expected(a[i], b[i], c[i], d[i], wa[i], wb[i], wc[i], wd[i]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicMixDslOps() {
  std::array<rund::kernel::i32, 8u> a{0x10000000, -2, 3, -4, 5, 6, 7, -8};
  std::array<rund::kernel::i32, 8u> b{0x08000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> c{0x04000000, -1, 5, 6, -3, 8, 2, 9};
  std::array<rund::kernel::i32, 8u> d{0x02000000, 7, 8, -5, 1, 4, -6, 3};
  std::array<rund::kernel::i32, 8u> wa{0x40000000, 3, -2, 4, 7, -6, 5, 2};
  std::array<rund::kernel::i32, 8u> wb{0x20000000, -6, 1, 7, -4, 3, 11, 2};
  std::array<rund::kernel::i32, 8u> wc{0x10000000, 5, -9, 4, 2, 8, -7, 6};
  std::array<rund::kernel::i32, 8u> wd{0x08000000, -7, 11,  -13,
                                       17,         19, -23, 29};
  TEST_ASSERT(RunCase(a, b, c, d, wa, wb, wc, wd));

  std::array<rund::kernel::i64, 8u> a64 = cpu::Widen(a);
  std::array<rund::kernel::i64, 8u> b64 = cpu::Widen(b);
  std::array<rund::kernel::i64, 8u> c64 = cpu::Widen(c);
  std::array<rund::kernel::i64, 8u> d64 = cpu::Widen(d);
  std::array<rund::kernel::i64, 8u> wa64 = cpu::Widen(wa);
  std::array<rund::kernel::i64, 8u> wb64 = cpu::Widen(wb);
  std::array<rund::kernel::i64, 8u> wc64 = cpu::Widen(wc);
  std::array<rund::kernel::i64, 8u> wd64 = cpu::Widen(wd);
  a64[0] = 0x1000000000000000ll;
  b64[0] = 0x0800000000000000ll;
  wa64[0] = 0x4000000000000000ll;
  wb64[0] = 0x2000000000000000ll;
  return RunCase(a64, b64, c64, d64, wa64, wb64, wc64, wd64);
}

} // namespace node_accel_contract
