#include "../../fixed.hpp"
#include "../../local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>

namespace node_accel_contract {
namespace {

inline constexpr rund::kernel::i32 kMax32 = 0x7fffffff;

[[nodiscard]] rund::kernel::i32 Sat32(const rund::kernel::i32 value) noexcept {
  return rund::math32::detail::ScalarClamp(value, 0, kMax32);
}

[[nodiscard]] rund::kernel::i32
Lerp32(const rund::kernel::i32 lhs, const rund::kernel::i32 rhs,
       const rund::kernel::i32 amount) noexcept {
  const rund::kernel::i32 t = Sat32(amount);
  return rund::math32::detail::ScalarAddSat(
      cpu::fixed::QuantizeProduct(
          lhs, rund::math32::detail::ScalarSubSat(kMax32, t)),
      cpu::fixed::QuantizeProduct(rhs, t));
}

[[nodiscard]] rund::kernel::i32
Bilerp32(const rund::kernel::i32 x00, const rund::kernel::i32 x10,
         const rund::kernel::i32 x01, const rund::kernel::i32 x11,
         const rund::kernel::i32 tx, const rund::kernel::i32 ty) noexcept {
  return Lerp32(Lerp32(x00, x10, tx), Lerp32(x01, x11, tx), ty);
}

[[nodiscard]] rund::kernel::i32
Expected(const rund::kernel::i32 mode, const rund::kernel::i32 x0,
         const rund::kernel::i32 x1, const rund::kernel::i32 x2,
         const rund::kernel::i32 x3, const rund::kernel::i32 t,
         const rund::kernel::i32 u) noexcept {
  if (mode == 3) {
    return Bilerp32(x0, x1, x2, x3, t, u);
  }
  if (mode == 2) {
    return Lerp32(x0, x1, t);
  }
  if (mode == 1) {
    return x1 < x0 ? 0 : kMax32;
  }
  return Sat32(t);
}

} // namespace

[[nodiscard]] bool RunsFixedLane32CompositeDslOps() {
  constexpr std::size_t kTileCount = 4u;
  std::array<rund::kernel::i32, kTileCount> mode{0, 1, 2, 3};
  std::array<rund::kernel::i32, kTileCount> x0{-1, 0x20000000, 0x10000000,
                                               0x08000000};
  std::array<rund::kernel::i32, kTileCount> x1{0x10000000, 0x10000000,
                                               0x30000000, 0x10000000};
  std::array<rund::kernel::i32, kTileCount> x2{0, 0, 0, 0x18000000};
  std::array<rund::kernel::i32, kTileCount> x3{0, 0, 0, 0x30000000};
  std::array<rund::kernel::i32, kTileCount> t{-0x70000000, 0, 0x40000000,
                                              0x40000000};
  std::array<rund::kernel::i32, kTileCount> u{0, 0, 0, 0x20000000};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"mode">(mode.data())
                        .read<"x0">(x0.data())
                        .read<"x1">(x1.data())
                        .read<"x2">(x2.data())
                        .read<"x3">(x3.data())
                        .read<"t">(t.data())
                        .read<"u">(u.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-composite-32")
          .on(body)
          .map([](auto i, auto b) {
            auto mode = b.template read<"mode">();
            auto x0 = b.template read<"x0">();
            auto x1 = b.template read<"x1">();
            auto x2 = b.template read<"x2">();
            auto x3 = b.template read<"x3">();
            auto t = b.template read<"t">();
            auto u = b.template read<"u">();
            auto out = b.template write<"out">();
            auto result = rund::compute_dsl::saturate(t[i]);
            result = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 3),
                rund::compute_dsl::lerp(x0[i], x1[i], x2[i], x3[i], t[i], u[i]),
                result);
            result = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 2),
                rund::compute_dsl::lerp(x0[i], x1[i], t[i]), result);
            out[i] = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 1),
                rund::compute_dsl::step(x0[i], x1[i]), result);
          });

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      29u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(out[index] == Expected(mode[index], x0[index], x1[index],
                                       x2[index], x3[index], t[index],
                                       u[index]));
  }
  return true;
}

} // namespace node_accel_contract
