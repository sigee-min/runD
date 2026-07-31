#include "../noise.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] bool RunNoiseGridOp(std::array<T, 4u> &x, std::array<T, 4u> &y,
                                  std::array<T, 4u> &tx, std::array<T, 4u> &ty,
                                  std::array<T, 4u> &seed,
                                  std::array<T, 4u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 63>()
          .template read<"x">(x.data())
          .template read<"y">(y.data())
          .template read<"tx">(tx.data())
          .template read<"ty">(ty.data())
          .template read<"seed">(seed.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(x.size())
          .template fixed<1, 31>()
          .template read<"x">(x.data())
          .template read<"y">(y.data())
          .template read<"tx">(tx.data())
          .template read<"ty">(ty.data())
          .template read<"seed">(seed.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-noise-grid")
          .on(body)
          .map([](auto i, auto b) {
            auto x = b.template read<"x">();
            auto y = b.template read<"y">();
            auto tx = b.template read<"tx">();
            auto ty = b.template read<"ty">();
            auto seed = b.template read<"seed">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::bit_xor(
                rund::compute_dsl::noise(x[i], y[i], tx[i], ty[i], seed[i]),
                rund::compute_dsl::noise(x[i], y[i], tx[i], ty[i]));
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(33u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

} // namespace

[[nodiscard]] bool RunsDeterministicNoiseGridDslOps() {
  std::array<rund::kernel::i32, 4u> x32{0, 1, -7, 0x12345678};
  std::array<rund::kernel::i32, 4u> y32{3, -5, 0x10000000, -0x10000000};
  std::array<rund::kernel::i32, 4u> tx32{-1, 0, 0x40000000, 0x7fffffff};
  std::array<rund::kernel::i32, 4u> ty32{0x7fffffff, 0x20000000, 0, -3};
  std::array<rund::kernel::i32, 4u> seed32{7, -11, 0x40000000, -0x20000000};
  std::array<rund::kernel::i32, 4u> out32{};
  TEST_ASSERT(RunNoiseGridOp(x32, y32, tx32, ty32, seed32, out32));
  for (std::size_t index = 0u; index < x32.size(); ++index) {
    TEST_ASSERT(out32[index] ==
                cpu::noise::XorBits(
                    cpu::noise::Noise(x32[index], y32[index], tx32[index],
                                      ty32[index], seed32[index]),
                    cpu::noise::Noise(x32[index], y32[index], tx32[index],
                                      ty32[index])));
  }

  std::array<rund::kernel::i64, 4u> x64{0, 1, -7, 0x123456789abcdefll};
  std::array<rund::kernel::i64, 4u> y64{3, -5, 0x1000000000000000ll,
                                        -0x1000000000000000ll};
  std::array<rund::kernel::i64, 4u> tx64{-1, 0, 0x4000000000000000ll,
                                         0x7fffffffffffffffll};
  std::array<rund::kernel::i64, 4u> ty64{0x7fffffffffffffffll,
                                         0x2000000000000000ll, 0, -3};
  std::array<rund::kernel::i64, 4u> seed64{17, -23, 0x4000000000000000ll,
                                           -0x2000000000000000ll};
  std::array<rund::kernel::i64, 4u> out64{};
  TEST_ASSERT(RunNoiseGridOp(x64, y64, tx64, ty64, seed64, out64));
  for (std::size_t index = 0u; index < x64.size(); ++index) {
    TEST_ASSERT(out64[index] ==
                cpu::noise::XorBits(
                    cpu::noise::Noise(x64[index], y64[index], tx64[index],
                                      ty64[index], seed64[index]),
                    cpu::noise::Noise(x64[index], y64[index], tx64[index],
                                      ty64[index])));
  }
  return true;
}

} // namespace node_accel_contract
