#include "fixed.hpp"
#include "stat.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T>
[[nodiscard]] T Expected(const T value, const T center, const T extra,
                         const T more) {
  return cpu::fixed::QuantizeSum<T>(
      cpu::stat::Half<T>(), cpu::stat::Third<T>(), cpu::stat::Quarter<T>(),
      cpu::stat::Mean(value, center), cpu::stat::Mean(value, center, extra),
      cpu::stat::Mean(value, center, extra, more),
      cpu::stat::Centered(value, center),
      cpu::stat::Centered(cpu::stat::CenteredOp::Abs, extra, center));
}

template <typename T>
[[nodiscard]] bool RunStatsOp(std::array<T, 8u> &value,
                              std::array<T, 8u> &center,
                              std::array<T, 8u> &extra, std::array<T, 8u> &more,
                              std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 63>()
          .template read<"value">(value.data())
          .template read<"center">(center.data())
          .template read<"extra">(extra.data())
          .template read<"more">(more.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 31>()
          .template read<"value">(value.data())
          .template read<"center">(center.data())
          .template read<"extra">(extra.data())
          .template read<"more">(more.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-stats")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto center = b.template read<"center">();
            auto extra = b.template read<"extra">();
            auto more = b.template read<"more">();
            auto out = b.template write<"out">();
            out[i] =
                rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Half,
                                         value[i]) +
                rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Third,
                                         value[i]) +
                rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Quarter,
                                         value[i]) +
                rund::compute_dsl::mean(value[i], center[i]) +
                rund::compute_dsl::mean(value[i], center[i], extra[i]) +
                rund::compute_dsl::mean(value[i], center[i], extra[i],
                                        more[i]) +
                rund::compute_dsl::centered(value[i], center[i]) +
                rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Abs,
                                            extra[i], center[i]);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const auto lanes = sizeof(T) == sizeof(rund::kernel::i64)
                         ? caps.fixed_lane64_lanes
                         : caps.fixed_lane32_lanes;
  const rund::kernel::BindingSet bindings =
      op.bindings<T>(61u, lanes, rund::kernel::ComputeApi::Metal);
  return rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings).ok;
}

template <typename T>
[[nodiscard]] bool RunCase(std::array<T, 8u> &value, std::array<T, 8u> &center,
                           std::array<T, 8u> &extra, std::array<T, 8u> &more) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunStatsOp(value, center, extra, more, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] == Expected(value[index], center[index],
                                       extra[index], more[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicStatsDslOps() {
  std::array<rund::kernel::i32, 8u> value32{-12, -4, 0, 5, 11, 19, -21, 7};
  std::array<rund::kernel::i32, 8u> center32{-10, -1, 1, 3, 9, 20, 3, 12};
  std::array<rund::kernel::i32, 8u> extra32{10, 6, 7, 5, 15, 30, -3, -8};
  std::array<rund::kernel::i32, 8u> more32{3, 4, -2, 9, 8, -11, 2, 1};
  TEST_ASSERT(RunCase(value32, center32, extra32, more32));

  std::array<rund::kernel::i64, 8u> value64{
      -12, -4, 0, 5, 0x1000000000000000ll, 19, -21, 7};
  std::array<rund::kernel::i64, 8u> center64{
      -10, -1, 1, 3, 0x0800000000000000ll, 20, 3, 12};
  std::array<rund::kernel::i64, 8u> extra64{10, 6,  7, 5, 0x1800000000000000ll,
                                            30, -3, -8};
  std::array<rund::kernel::i64, 8u> more64{3,   4, -2, 9, 0x0400000000000000ll,
                                           -11, 2, 1};
  return RunCase(value64, center64, extra64, more64);
}

} // namespace node_accel_contract
