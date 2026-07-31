#include "cross.hpp"
#include <node/accel/cpu/simd.hpp>

#include "fixed.hpp"

#include <array>
#include <iostream>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] bool RunCrossOp(std::array<T, 4u> &ax, std::array<T, 4u> &ay,
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
      rund::compute_dsl::def("node-cpu-simd-cross")
          .on(body)
          .map([](auto i, auto b) {
            auto ax = b.template read<"ax">();
            auto ay = b.template read<"ay">();
            auto az = b.template read<"az">();
            auto bx = b.template read<"bx">();
            auto by = b.template read<"by">();
            auto bz = b.template read<"bz">();
            auto out = b.template write<"out">();
            out[i] =
                rund::compute_dsl::cross(ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::cross(rund::compute_dsl::Axis::X, ax[i],
                                         ay[i], az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::cross(rund::compute_dsl::Axis::Y, ax[i],
                                         ay[i], az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::cross(rund::compute_dsl::Axis::Z, ax[i],
                                         ay[i], az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::reject(rund::compute_dsl::Axis::X, ax[i],
                                          ay[i], bx[i], by[i]) +
                rund::compute_dsl::reject(rund::compute_dsl::Axis::Y, ax[i],
                                          ay[i], bx[i], by[i]) +
                rund::compute_dsl::reject(rund::compute_dsl::Axis::X, ax[i],
                                          ay[i], az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::reject(rund::compute_dsl::Axis::Y, ax[i],
                                          ay[i], az[i], bx[i], by[i], bz[i]) +
                rund::compute_dsl::reject(rund::compute_dsl::Axis::Z, ax[i],
                                          ay[i], az[i], bx[i], by[i], bz[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(47u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

[[nodiscard]] rund::kernel::i32
Expected32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
           const rund::kernel::i32 az, const rund::kernel::i32 bx,
           const rund::kernel::i32 by, const rund::kernel::i32 bz) {
  rund::kernel::i128 out = cpu::cross::Cross2_32(ax, ay, bx, by);
  out += cpu::cross::Cross3X_32(ax, ay, az, bx, by, bz);
  out += cpu::cross::Cross3Y_32(ax, ay, az, bx, by, bz);
  out += cpu::cross::Cross3Z_32(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject2X_32(ax, ay, bx, by);
  out += cpu::cross::Reject2Y_32(ax, ay, bx, by);
  out += cpu::cross::Reject3X_32(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject3Y_32(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject3Z_32(ax, ay, az, bx, by, bz);
  return static_cast<rund::kernel::i32>(
      out < std::numeric_limits<rund::kernel::i32>::min()
          ? std::numeric_limits<rund::kernel::i32>::min()
      : out > std::numeric_limits<rund::kernel::i32>::max()
          ? std::numeric_limits<rund::kernel::i32>::max()
          : out);
}

[[nodiscard]] rund::kernel::i64
Expected64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
           const rund::kernel::i64 az, const rund::kernel::i64 bx,
           const rund::kernel::i64 by, const rund::kernel::i64 bz) {
  rund::kernel::i128 out = cpu::cross::Cross2_64(ax, ay, bx, by);
  out += cpu::cross::Cross3X_64(ax, ay, az, bx, by, bz);
  out += cpu::cross::Cross3Y_64(ax, ay, az, bx, by, bz);
  out += cpu::cross::Cross3Z_64(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject2X_64(ax, ay, bx, by);
  out += cpu::cross::Reject2Y_64(ax, ay, bx, by);
  out += cpu::cross::Reject3X_64(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject3Y_64(ax, ay, az, bx, by, bz);
  out += cpu::cross::Reject3Z_64(ax, ay, az, bx, by, bz);
  return static_cast<rund::kernel::i64>(
      out < std::numeric_limits<rund::kernel::i64>::min()
          ? std::numeric_limits<rund::kernel::i64>::min()
      : out > std::numeric_limits<rund::kernel::i64>::max()
          ? std::numeric_limits<rund::kernel::i64>::max()
          : out);
}

} // namespace

[[nodiscard]] bool RunsDeterministicCrossDslOps() {
  std::array<rund::kernel::i32, 4u> ax32{0x20000000, -0x20000000, 0x10000000,
                                         0};
  std::array<rund::kernel::i32, 4u> ay32{0x10000000, 0x20000000, -0x10000000,
                                         0};
  std::array<rund::kernel::i32, 4u> az32{-0x10000000, 0x10000000, 0x20000000,
                                         0};
  std::array<rund::kernel::i32, 4u> bx32{0x30000000, 0x10000000, -0x20000000,
                                         0};
  std::array<rund::kernel::i32, 4u> by32{-0x10000000, 0x20000000, 0x10000000,
                                         0};
  std::array<rund::kernel::i32, 4u> bz32{0x10000000, -0x10000000, 0x30000000,
                                         0};
  std::array<rund::kernel::i32, 4u> out32{};
  TEST_ASSERT(RunCrossOp(ax32, ay32, az32, bx32, by32, bz32, out32));
  for (std::size_t i = 0u; i < out32.size(); ++i) {
    TEST_ASSERT(out32[i] == Expected32(ax32[i], ay32[i], az32[i], bx32[i],
                                       by32[i], bz32[i]));
  }

  std::array<rund::kernel::i64, 4u> ax64{
      0x2000000000000000ll, -0x2000000000000000ll, 0x1000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> ay64{
      0x1000000000000000ll, 0x2000000000000000ll, -0x1000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> az64{
      -0x1000000000000000ll, 0x1000000000000000ll, 0x2000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> bx64{
      0x3000000000000000ll, 0x1000000000000000ll, -0x2000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> by64{
      -0x1000000000000000ll, 0x2000000000000000ll, 0x1000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> bz64{
      0x1000000000000000ll, -0x1000000000000000ll, 0x3000000000000000ll, 0};
  std::array<rund::kernel::i64, 4u> out64{};
  TEST_ASSERT(RunCrossOp(ax64, ay64, az64, bx64, by64, bz64, out64));
  for (std::size_t i = 0u; i < out64.size(); ++i) {
    if (out64[i] !=
        Expected64(ax64[i], ay64[i], az64[i], bx64[i], by64[i], bz64[i])) {
      std::cerr << "cross64 mismatch index=" << i << " actual=" << out64[i]
                << " expected="
                << Expected64(ax64[i], ay64[i], az64[i], bx64[i], by64[i],
                              bz64[i])
                << '\n';
    }
    TEST_ASSERT(out64[i] == Expected64(ax64[i], ay64[i], az64[i], bx64[i],
                                       by64[i], bz64[i]));
  }
  return true;
}

} // namespace node_accel_contract
