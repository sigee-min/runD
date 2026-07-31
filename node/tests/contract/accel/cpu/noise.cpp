#include "noise.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstdio>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] bool RunNoiseOp(std::array<T, 4u> &cell, std::array<T, 4u> &t,
                              std::array<T, 4u> &seed, std::array<T, 4u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(cell.size())
          .template fixed<1, 63>()
          .template read<"cell">(cell.data())
          .template read<"t">(t.data())
          .template read<"seed">(seed.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(cell.size())
          .template fixed<1, 31>()
          .template read<"cell">(cell.data())
          .template read<"t">(t.data())
          .template read<"seed">(seed.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-noise")
          .on(body)
          .map([](auto i, auto b) {
            auto cell = b.template read<"cell">();
            auto t = b.template read<"t">();
            auto seed = b.template read<"seed">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::bit_xor(
                rund::compute_dsl::noise(cell[i], t[i], seed[i]),
                rund::compute_dsl::noise(cell[i], t[i]));
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(32u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

} // namespace

[[nodiscard]] bool RunsDeterministicNoiseDslOps() {
  std::array<rund::kernel::i32, 4u> cell32{0, 1, -7, 0x12345678};
  std::array<rund::kernel::i32, 4u> t32{-1, 0, 0x40000000, 0x7fffffff};
  std::array<rund::kernel::i32, 4u> seed32{7, -11, 0x40000000, -0x20000000};
  std::array<rund::kernel::i32, 4u> out32{};
  TEST_ASSERT(RunNoiseOp(cell32, t32, seed32, out32));
  for (std::size_t index = 0u; index < cell32.size(); ++index) {
    const auto expected = cpu::noise::XorBits(
        cpu::noise::Noise(cell32[index], t32[index], seed32[index]),
        cpu::noise::Noise(cell32[index], t32[index]));
    if (out32[index] != expected) {
      std::fprintf(stderr,
                   "noise32 mismatch index=%zu cell=%d amount=%d seed=%d "
                   "actual=%d expected=%d\n",
                   index, cell32[index], t32[index], seed32[index],
                   out32[index], expected);
    }
    TEST_ASSERT(out32[index] == expected);
  }

  std::array<rund::kernel::i64, 4u> cell64{0, 1, -7, 0x123456789abcdefll};
  std::array<rund::kernel::i64, 4u> t64{-1, 0, 0x4000000000000000ll,
                                        0x7fffffffffffffffll};
  std::array<rund::kernel::i64, 4u> seed64{17, -23, 0x4000000000000000ll,
                                           -0x2000000000000000ll};
  std::array<rund::kernel::i64, 4u> out64{};
  TEST_ASSERT(RunNoiseOp(cell64, t64, seed64, out64));
  for (std::size_t index = 0u; index < cell64.size(); ++index) {
    const auto expected = cpu::noise::XorBits(
        cpu::noise::Noise(cell64[index], t64[index], seed64[index]),
        cpu::noise::Noise(cell64[index], t64[index]));
    if (out64[index] != expected) {
      std::fprintf(stderr,
                   "noise64 mismatch index=%zu cell=%lld amount=%lld "
                   "seed=%lld actual=%lld expected=%lld\n",
                   index, static_cast<long long>(cell64[index]),
                   static_cast<long long>(t64[index]),
                   static_cast<long long>(seed64[index]),
                   static_cast<long long>(out64[index]),
                   static_cast<long long>(expected));
    }
    TEST_ASSERT(out64[index] == expected);
  }
  return true;
}

} // namespace node_accel_contract
