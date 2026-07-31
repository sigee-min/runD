#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool CpuSimdFixedLane32OrderingOpsCovered() {
  constexpr std::size_t kTileCount = 8u;
  std::array<rund::kernel::i32, kTileCount> lhs{-9, 4,  12,   -30,
                                                7,  99, -128, 64};
  std::array<rund::kernel::i32, kTileCount> rhs{3,  4,   -8,   -10,
                                                21, -50, -128, 16};
  std::array<rund::kernel::i32, kTileCount> value{-20, 4,  0,    -40,
                                                  30,  10, -127, 80};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"lhs">(lhs.data())
                        .read<"rhs">(rhs.data())
                        .read<"value">(value.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-ordering-32")
          .on(body)
          .map([](auto i, auto b) {
            auto lhs = b.template read<"lhs">();
            auto rhs = b.template read<"rhs">();
            auto value = b.template read<"value">();
            auto out = b.template write<"out">();

            const auto low = rund::compute_dsl::min(lhs[i], rhs[i]);
            const auto high = rund::compute_dsl::max(lhs[i], rhs[i]);
            const auto bounded = rund::compute_dsl::clamp(value[i], low, high);
            const auto lower_side = rund::compute_dsl::select(
                rund::compute_dsl::lt(value[i], low), low, bounded);
            const auto final = rund::compute_dsl::select(
                rund::compute_dsl::le(value[i], high), lower_side, high);
            out[i] = final + low + high;
          });

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      27u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    const rund::kernel::i32 low =
        rund::math32::detail::ScalarMin(lhs[index], rhs[index]);
    const rund::kernel::i32 high =
        rund::math32::detail::ScalarMax(lhs[index], rhs[index]);
    const rund::kernel::i32 bounded =
        rund::math32::detail::ScalarClamp(value[index], low, high);
    const rund::kernel::i32 lower_side = value[index] < low ? low : bounded;
    const rund::kernel::i32 final = value[index] <= high ? lower_side : high;
    const rund::kernel::i32 expected = rund::math32::detail::ScalarAddWrap(
        rund::math32::detail::ScalarAddWrap(final, low), high);
    TEST_ASSERT(out[index] == expected);
  }
  return true;
}

} // namespace node_accel_contract
