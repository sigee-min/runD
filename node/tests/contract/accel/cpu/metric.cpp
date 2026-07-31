#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <limits>

namespace node_accel_contract {
namespace {

template <typename T> [[nodiscard]] T Abs(const T value) noexcept {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAbs(value);
  } else {
    return rund::math32::detail::ScalarAbs(value);
  }
}

template <typename T> [[nodiscard]] T AddSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarAddSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarAddSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T SubSat(const T lhs, const T rhs) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
    return rund::math64::detail::ScalarSubSat(lhs, rhs);
  } else {
    return rund::math32::detail::ScalarSubSat(lhs, rhs);
  }
}

template <typename T> [[nodiscard]] T Max(const T lhs, const T rhs) {
  return lhs > rhs ? lhs : rhs;
}

template <typename T> [[nodiscard]] T AbsDiff(const T lhs, const T rhs) {
  return Abs(SubSat(lhs, rhs));
}

template <typename T> [[nodiscard]] T L1(const T x, const T y) {
  return AddSat(Abs(x), Abs(y));
}

template <typename T> [[nodiscard]] T L1(const T x, const T y, const T z) {
  return AddSat(L1(x, y), Abs(z));
}

template <typename T> [[nodiscard]] T Linf(const T x, const T y) {
  return Max(Abs(x), Abs(y));
}

template <typename T> [[nodiscard]] T Linf(const T x, const T y, const T z) {
  return Max(Linf(x, y), Abs(z));
}

template <typename T>
[[nodiscard]] T Expected(const T ax, const T ay, const T az, const T bx,
                         const T by, const T bz) {
  const T dx = SubSat(ax, bx);
  const T dy = SubSat(ay, by);
  const T dz = SubSat(az, bz);
  rund::kernel::i128 result = AbsDiff(ax, bx);
  result += L1(ax, ay);
  result += L1(ax, ay, az);
  result += Linf(ax, ay);
  result += Linf(ax, ay, az);
  result += L1(dx, dy);
  result += L1(dx, dy, dz);
  result += Linf(dx, dy);
  result += Linf(dx, dy, dz);
  const rund::kernel::i128 lower = std::numeric_limits<T>::min();
  const rund::kernel::i128 upper = std::numeric_limits<T>::max();
  return static_cast<T>(result < lower   ? lower
                        : result > upper ? upper
                                         : result);
}

template <typename T>
[[nodiscard]] bool RunMetricOp(std::array<T, 4u> &ax, std::array<T, 4u> &ay,
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
      rund::compute_dsl::def("node-cpu-simd-metric")
          .on(body)
          .map([](auto i, auto b) {
            auto ax = b.template read<"ax">();
            auto ay = b.template read<"ay">();
            auto az = b.template read<"az">();
            auto bx = b.template read<"bx">();
            auto by = b.template read<"by">();
            auto bz = b.template read<"bz">();
            auto out = b.template write<"out">();
            using Norm = rund::compute_dsl::Norm;
            out[i] =
                rund::compute_dsl::absdiff(ax[i], bx[i]) +
                rund::compute_dsl::len(Norm::L1, ax[i], ay[i]) +
                rund::compute_dsl::len(Norm::L1, ax[i], ay[i], az[i]) +
                rund::compute_dsl::len(Norm::LInf, ax[i], ay[i]) +
                rund::compute_dsl::len(Norm::LInf, ax[i], ay[i], az[i]) +
                rund::compute_dsl::dist(Norm::L1, ax[i], ay[i], bx[i], by[i]) +
                rund::compute_dsl::dist(Norm::L1, ax[i], ay[i], az[i], bx[i],
                                        by[i], bz[i]) +
                rund::compute_dsl::dist(Norm::LInf, ax[i], ay[i], bx[i],
                                        by[i]) +
                rund::compute_dsl::dist(Norm::LInf, ax[i], ay[i], az[i], bx[i],
                                        by[i], bz[i]);
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

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 4u> &ax, std::array<T, 4u> &ay,
                           std::array<T, 4u> &az, std::array<T, 4u> &bx,
                           std::array<T, 4u> &by, std::array<T, 4u> &bz) {
  std::array<T, 4u> out{};
  TEST_ASSERT(RunMetricOp(ax, ay, az, bx, by, bz, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] == Expected(ax[index], ay[index], az[index],
                                       bx[index], by[index], bz[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicMetricDslOps() {
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
  TEST_ASSERT(RunCase(ax32, ay32, az32, bx32, by32, bz32));

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
  return RunCase(ax64, ay64, az64, bx64, by64, bz64);
}

} // namespace node_accel_contract
