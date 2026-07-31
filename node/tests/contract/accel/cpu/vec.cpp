#include "vec.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] bool RunVecOp(std::array<T, 4u> &ax, std::array<T, 4u> &ay,
                            std::array<T, 4u> &az, std::array<T, 4u> &bx,
                            std::array<T, 4u> &by, std::array<T, 4u> &bz,
                            std::array<T, 4u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(ax.size())
          .template fixed<1, 63>()
          .template read<"ax">(ax.data())
          .template read<"ay">(ay.data())
          .template read<"az">(az.data())
          .template read<"bx">(bx.data())
          .template read<"by">(by.data())
          .template read<"bz">(bz.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(ax.size())
          .template fixed<1, 31>()
          .template read<"ax">(ax.data())
          .template read<"ay">(ay.data())
          .template read<"az">(az.data())
          .template read<"bx">(bx.data())
          .template read<"by">(by.data())
          .template read<"bz">(bz.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-vec")
          .on(body)
          .map([](auto i, auto b) {
            auto ax = b.template read<"ax">();
            auto ay = b.template read<"ay">();
            auto az = b.template read<"az">();
            auto bx = b.template read<"bx">();
            auto by = b.template read<"by">();
            auto bz = b.template read<"bz">();
            auto out = b.template write<"out">();
            out[i] = rund::compute_dsl::dot(ax[i], ay[i], bx[i], by[i]) +
                     rund::compute_dsl::dot(ax[i], ay[i], az[i], bx[i], by[i],
                                            bz[i]) +
                     rund::compute_dsl::len(ax[i], ay[i]) +
                     rund::compute_dsl::len(ax[i], ay[i], az[i]) +
                     rund::compute_dsl::dist(ax[i], ay[i], bx[i], by[i]) +
                     rund::compute_dsl::dist(ax[i], ay[i], az[i], bx[i], by[i],
                                             bz[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(41u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

[[nodiscard]] rund::kernel::i32
Expected32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
           const rund::kernel::i32 az, const rund::kernel::i32 bx,
           const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  std::int64_t result = cpu::vec::Dot2_32(ax, ay, bx, by);
  result += cpu::vec::Dot3_32(ax, ay, az, bx, by, bz);
  result += cpu::vec::Len2_32(ax, ay);
  result += cpu::vec::Len3_32(ax, ay, az);
  result += cpu::vec::Dist2_32(ax, ay, bx, by);
  result += cpu::vec::Dist3_32(ax, ay, az, bx, by, bz);
  return static_cast<rund::kernel::i32>(
      result < std::numeric_limits<rund::kernel::i32>::min()
          ? std::numeric_limits<rund::kernel::i32>::min()
      : result > std::numeric_limits<rund::kernel::i32>::max()
          ? std::numeric_limits<rund::kernel::i32>::max()
          : result);
}

[[nodiscard]] rund::kernel::i64
Expected64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
           const rund::kernel::i64 az, const rund::kernel::i64 bx,
           const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  rund::kernel::i128 result = cpu::vec::Dot2_64(ax, ay, bx, by);
  result += cpu::vec::Dot3_64(ax, ay, az, bx, by, bz);
  result += cpu::vec::Len2_64(ax, ay);
  result += cpu::vec::Len3_64(ax, ay, az);
  result += cpu::vec::Dist2_64(ax, ay, bx, by);
  result += cpu::vec::Dist3_64(ax, ay, az, bx, by, bz);
  const rund::kernel::i128 lower = std::numeric_limits<rund::kernel::i64>::min();
  const rund::kernel::i128 upper = std::numeric_limits<rund::kernel::i64>::max();
  return static_cast<rund::kernel::i64>(result < lower   ? lower
                                        : result > upper ? upper
                                                         : result);
}

} // namespace

[[nodiscard]] bool RunsDeterministicVecDslOps() {
  std::array<rund::kernel::i32, 4u> ax32{0x40000000, -0x20000000, 0x7fffffff,
                                         -0x40000000};
  std::array<rund::kernel::i32, 4u> ay32{0, 0x30000000, -0x10000000,
                                         0x40000000};
  std::array<rund::kernel::i32, 4u> az32{0x10000000, -0x10000000, 0x20000000,
                                         0};
  std::array<rund::kernel::i32, 4u> bx32{0x20000000, 0x10000000, -0x70000000,
                                         0x40000000};
  std::array<rund::kernel::i32, 4u> by32{-0x10000000, 0x20000000, 0x10000000,
                                         -0x40000000};
  std::array<rund::kernel::i32, 4u> bz32{0, 0x30000000, -0x20000000,
                                         0x10000000};
  std::array<rund::kernel::i32, 4u> out32{};
  TEST_ASSERT(RunVecOp(ax32, ay32, az32, bx32, by32, bz32, out32));
  for (std::size_t index = 0u; index < out32.size(); ++index) {
    TEST_ASSERT(out32[index] == Expected32(ax32[index], ay32[index],
                                           az32[index], bx32[index],
                                           by32[index], bz32[index]));
  }

  std::array<rund::kernel::i64, 4u> ax64{
      0x4000000000000000ll, -0x2000000000000000ll, 0x7fffffffffffffffll,
      -0x4000000000000000ll};
  std::array<rund::kernel::i64, 4u> ay64{
      0, 0x3000000000000000ll, -0x1000000000000000ll, 0x4000000000000000ll};
  std::array<rund::kernel::i64, 4u> az64{
      0x1000000000000000ll, -0x1000000000000000ll, 0x2000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> bx64{
      0x2000000000000000ll, 0x1000000000000000ll, -0x7000000000000000ll,
      0x4000000000000000ll};
  std::array<rund::kernel::i64, 4u> by64{
      -0x1000000000000000ll, 0x2000000000000000ll, 0x1000000000000000ll,
      -0x4000000000000000ll};
  std::array<rund::kernel::i64, 4u> bz64{
      0, 0x3000000000000000ll, -0x2000000000000000ll, 0x1000000000000000ll};
  std::array<rund::kernel::i64, 4u> out64{};
  TEST_ASSERT(RunVecOp(ax64, ay64, az64, bx64, by64, bz64, out64));
  for (std::size_t index = 0u; index < out64.size(); ++index) {
    TEST_ASSERT(out64[index] == Expected64(ax64[index], ay64[index],
                                           az64[index], bx64[index],
                                           by64[index], bz64[index]));
  }
  return true;
}

} // namespace node_accel_contract
