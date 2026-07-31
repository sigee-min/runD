#include "fixed.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] bool RunNormOp(std::array<T, 5u> &in_lo, std::array<T, 5u> &in_hi,
                             std::array<T, 5u> &out_lo,
                             std::array<T, 5u> &out_hi,
                             std::array<T, 5u> &value, std::array<T, 5u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(in_lo.size())
          .template fixed<1, 63>()
          .template read<"in_lo">(in_lo.data())
          .template read<"in_hi">(in_hi.data())
          .template read<"out_lo">(out_lo.data())
          .template read<"out_hi">(out_hi.data())
          .template read<"value">(value.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(in_lo.size())
          .template fixed<1, 31>()
          .template read<"in_lo">(in_lo.data())
          .template read<"in_hi">(in_hi.data())
          .template read<"out_lo">(out_lo.data())
          .template read<"out_hi">(out_hi.data())
          .template read<"value">(value.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-norm")
          .on(body)
          .map([](auto i, auto b) {
            auto in_lo = b.template read<"in_lo">();
            auto in_hi = b.template read<"in_hi">();
            auto out_lo = b.template read<"out_lo">();
            auto out_hi = b.template read<"out_hi">();
            auto value = b.template read<"value">();
            auto out = b.template write<"out">();
            out[i] =
                rund::compute_dsl::remap(in_lo[i], in_hi[i], out_lo[i],
                                         out_hi[i], value[i]) +
                rund::compute_dsl::unlerp(in_lo[i], in_hi[i], value[i]) +
                rund::compute_dsl::smoothstep(in_lo[i], in_hi[i], value[i]) +
                rund::compute_dsl::smootherstep(in_lo[i], in_hi[i], value[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(37u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

[[nodiscard]] rund::kernel::i32
Expected32(const rund::kernel::i32 in_lo, const rund::kernel::i32 in_hi,
           const rund::kernel::i32 out_lo, const rund::kernel::i32 out_hi,
           const rund::kernel::i32 value) noexcept {
  const auto sum = static_cast<rund::kernel::i64>(cpu::fixed::Remap32(
                       in_lo, in_hi, out_lo, out_hi, value)) +
                   cpu::fixed::Unlerp32(in_lo, in_hi, value) +
                   cpu::fixed::Smoothstep32(in_lo, in_hi, value) +
                   cpu::fixed::Smootherstep32(in_lo, in_hi, value);
  return sum > std::numeric_limits<rund::kernel::i32>::max()
             ? std::numeric_limits<rund::kernel::i32>::max()
         : sum < std::numeric_limits<rund::kernel::i32>::min()
             ? std::numeric_limits<rund::kernel::i32>::min()
             : static_cast<rund::kernel::i32>(sum);
}

[[nodiscard]] rund::kernel::i64
Expected64(const rund::kernel::i64 in_lo, const rund::kernel::i64 in_hi,
           const rund::kernel::i64 out_lo, const rund::kernel::i64 out_hi,
           const rund::kernel::i64 value) noexcept {
  const auto sum = static_cast<rund::kernel::i128>(cpu::fixed::Remap64(
                       in_lo, in_hi, out_lo, out_hi, value)) +
                   cpu::fixed::Unlerp64(in_lo, in_hi, value) +
                   cpu::fixed::Smoothstep64(in_lo, in_hi, value) +
                   cpu::fixed::Smootherstep64(in_lo, in_hi, value);
  return sum > std::numeric_limits<rund::kernel::i64>::max()
             ? std::numeric_limits<rund::kernel::i64>::max()
         : sum < std::numeric_limits<rund::kernel::i64>::min()
             ? std::numeric_limits<rund::kernel::i64>::min()
             : static_cast<rund::kernel::i64>(sum);
}

} // namespace

[[nodiscard]] bool RunsDeterministicNormDslOps() {
  std::array<rund::kernel::i32, 5u> in_lo32{0, 0x10000000, 0x40000000,
                                            0x30000000, 0x40000000};
  std::array<rund::kernel::i32, 5u> in_hi32{0x7fffffff, 0x50000000, 0x40000000,
                                            0x10000000, 0x60000000};
  std::array<rund::kernel::i32, 5u> out_lo32{0, -0x10000000, 0x20000000,
                                             0x10000000, 0x08000000};
  std::array<rund::kernel::i32, 5u> out_hi32{0x7fffffff, 0x40000000, 0x60000000,
                                             0x70000000, 0x20000000};
  std::array<rund::kernel::i32, 5u> value32{-1, 0x30000000, 0x50000000,
                                            0x20000000, 0x70000000};
  std::array<rund::kernel::i32, 5u> out32{};
  TEST_ASSERT(RunNormOp(in_lo32, in_hi32, out_lo32, out_hi32, value32, out32));
  for (std::size_t index = 0u; index < out32.size(); ++index) {
    TEST_ASSERT(out32[index] == Expected32(in_lo32[index], in_hi32[index],
                                           out_lo32[index], out_hi32[index],
                                           value32[index]));
  }

  std::array<rund::kernel::i64, 5u> in_lo64{
      0, 0x1000000000000000ll, 0x4000000000000000ll, 0x3000000000000000ll,
      0x4000000000000000ll};
  std::array<rund::kernel::i64, 5u> in_hi64{
      0x7fffffffffffffffll, 0x5000000000000000ll, 0x4000000000000000ll,
      0x1000000000000000ll, 0x6000000000000000ll};
  std::array<rund::kernel::i64, 5u> out_lo64{
      0, -0x1000000000000000ll, 0x2000000000000000ll, 0x1000000000000000ll,
      0x0800000000000000ll};
  std::array<rund::kernel::i64, 5u> out_hi64{
      0x7fffffffffffffffll, 0x4000000000000000ll, 0x6000000000000000ll,
      0x7000000000000000ll, 0x2000000000000000ll};
  std::array<rund::kernel::i64, 5u> value64{
      -1, 0x3000000000000000ll, 0x5000000000000000ll, 0x2000000000000000ll,
      0x7000000000000000ll};
  std::array<rund::kernel::i64, 5u> out64{};
  TEST_ASSERT(RunNormOp(in_lo64, in_hi64, out_lo64, out_hi64, value64, out64));
  for (std::size_t index = 0u; index < out64.size(); ++index) {
    TEST_ASSERT(out64[index] == Expected64(in_lo64[index], in_hi64[index],
                                           out_lo64[index], out_hi64[index],
                                           value64[index]));
  }
  return true;
}

} // namespace node_accel_contract
