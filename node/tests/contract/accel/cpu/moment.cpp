#include "fixed.hpp"
#include "stat.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Expected(const T a, const T b, const T c, const T d) {
  return cpu::fixed::QuantizeSum<T>(
      cpu::stat::Centered(cpu::stat::CenteredOp::Squared, a, b),
      cpu::stat::Var(a, b), cpu::stat::Var(a, b, c), cpu::stat::Var(a, b, c, d),
      cpu::stat::Rms(a, b), cpu::stat::Rms(a, b, c),
      cpu::stat::Rms(a, b, c, d));
}

template <typename T>
[[nodiscard]] bool RunMomentOp(std::array<T, 8u> &a, std::array<T, 8u> &b,
                               std::array<T, 8u> &c, std::array<T, 8u> &d,
                               std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(a.size())
          .template fixed<1, 63>()
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template read<"d">(d.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(a.size())
          .template fixed<1, 31>()
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template read<"d">(d.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-moment")
          .on(body)
          .map([](auto i, auto bind) {
            auto a = bind.template read<"a">();
            auto b = bind.template read<"b">();
            auto c = bind.template read<"c">();
            auto d = bind.template read<"d">();
            auto out = bind.template write<"out">();
            out[i] = rund::compute_dsl::centered(
                         rund::compute_dsl::CenteredOp::Squared, a[i], b[i]) +
                     rund::compute_dsl::var(a[i], b[i]) +
                     rund::compute_dsl::var(a[i], b[i], c[i]) +
                     rund::compute_dsl::var(a[i], b[i], c[i], d[i]) +
                     rund::compute_dsl::rms(a[i], b[i]) +
                     rund::compute_dsl::rms(a[i], b[i], c[i]) +
                     rund::compute_dsl::rms(a[i], b[i], c[i], d[i]);
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
[[nodiscard]] bool RunCase(std::array<T, 8u> &a, std::array<T, 8u> &b,
                           std::array<T, 8u> &c, std::array<T, 8u> &d) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunMomentOp(a, b, c, d, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] == Expected(a[index], b[index], c[index], d[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicMomentDslOps() {
  std::array<rund::kernel::i32, 8u> a32{-12, -4, 0, 5, 11, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> b32{-10, -1, 1, 3, 9, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> c32{10, 6, 7, 5, 15, 30, -3, -8};
  std::array<rund::kernel::i32, 8u> d32{3, 4, -2, 9, 8, -11, 2, 1};
  TEST_ASSERT(RunCase(a32, b32, c32, d32));

  std::array<rund::kernel::i64, 8u> a64{-12, -4,  0, 5, 0x1000000000000000ll,
                                        19,  -21, 7};
  std::array<rund::kernel::i64, 8u> b64{-10, -1, 1, 3, 0x0800000000000000ll,
                                        20,  3,  12};
  std::array<rund::kernel::i64, 8u> c64{10, 6,  7, 5, 0x1800000000000000ll,
                                        30, -3, -8};
  std::array<rund::kernel::i64, 8u> d64{3,   4, -2, 9, 0x0400000000000000ll,
                                        -11, 2, 1};
  return RunCase(a64, b64, c64, d64);
}

} // namespace node_accel_contract
