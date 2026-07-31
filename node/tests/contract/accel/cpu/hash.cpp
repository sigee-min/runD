#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <bit>

namespace node_accel_contract {
namespace {

[[nodiscard]] rund::kernel::u32 Hash32Bits(rund::kernel::u32 value) noexcept {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

[[nodiscard]] rund::kernel::u64 Hash64Bits(rund::kernel::u64 value) noexcept {
  value ^= value >> 33u;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33u;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33u;
  return value;
}

[[nodiscard]] rund::kernel::i32
Expected32(const rund::kernel::i32 value,
           const rund::kernel::i32 seed) noexcept {
  const rund::kernel::u32 bits = std::bit_cast<rund::kernel::u32>(value);
  const rund::kernel::u32 seed_bits = std::bit_cast<rund::kernel::u32>(seed);
  const rund::kernel::u32 mixed = Hash32Bits(bits ^ seed_bits);
  const rund::kernel::u32 unit = Hash32Bits(bits) & 0x7fffffffu;
  const rund::kernel::u32 unit_seed = mixed & 0x7fffffffu;
  return std::bit_cast<rund::kernel::i32>(mixed ^ unit ^ unit_seed);
}

[[nodiscard]] rund::kernel::i64
Expected64(const rund::kernel::i64 value,
           const rund::kernel::i64 seed) noexcept {
  const rund::kernel::u64 bits = std::bit_cast<rund::kernel::u64>(value);
  const rund::kernel::u64 seed_bits = std::bit_cast<rund::kernel::u64>(seed);
  const rund::kernel::u64 mixed = Hash64Bits(bits ^ seed_bits);
  const rund::kernel::u64 unit = Hash64Bits(bits) & 0x7fffffffffffffffull;
  const rund::kernel::u64 unit_seed = mixed & 0x7fffffffffffffffull;
  return std::bit_cast<rund::kernel::i64>(mixed ^ unit ^ unit_seed);
}

template <typename T>
[[nodiscard]] bool RunHashOp(std::array<T, 4u> &values,
                             std::array<T, 4u> &seeds, std::array<T, 4u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(values.size())
          .template fixed<1, 63>()
          .template read<"value">(values.data())
          .template read<"seed">(seeds.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(values.size())
          .template fixed<1, 31>()
          .template read<"value">(values.data())
          .template read<"seed">(seeds.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-hash")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto seed = b.template read<"seed">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::bit_xor(
                rund::compute_dsl::hash(value[i], seed[i]),
                rund::compute_dsl::bit_xor(
                    rund::compute_dsl::hash(rund::compute_dsl::HashOp::Unit,
                                            value[i]),
                    rund::compute_dsl::hash(rund::compute_dsl::HashOp::Unit,
                                            value[i], seed[i])));
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(31u, lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);
  return run.ok;
}

} // namespace

[[nodiscard]] bool RunsDeterministicHashDslOps() {
  std::array<rund::kernel::i32, 4u> values32{0, 1, -1, 0x12345678};
  std::array<rund::kernel::i32, 4u> seeds32{7, -11, 0x40000000, -0x20000000};
  std::array<rund::kernel::i32, 4u> out32{};
  TEST_ASSERT(RunHashOp(values32, seeds32, out32));
  for (std::size_t index = 0u; index < values32.size(); ++index) {
    TEST_ASSERT(out32[index] == Expected32(values32[index], seeds32[index]));
  }

  std::array<rund::kernel::i64, 4u> values64{0, 1, -1, 0x123456789abcdefll};
  std::array<rund::kernel::i64, 4u> seeds64{17, -23, 0x4000000000000000ll,
                                            -0x2000000000000000ll};
  std::array<rund::kernel::i64, 4u> out64{};
  TEST_ASSERT(RunHashOp(values64, seeds64, out64));
  for (std::size_t index = 0u; index < values64.size(); ++index) {
    TEST_ASSERT(out64[index] == Expected64(values64[index], seeds64[index]));
  }
  return true;
}

} // namespace node_accel_contract
