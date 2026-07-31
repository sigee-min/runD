#include "fixed.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <cstddef>

namespace node_accel_contract {
namespace {

template <typename T> [[nodiscard]] T Bool(const bool value) {
  return value ? T{1} : T{0};
}

template <typename T>
[[nodiscard]] T Expected(const T value, const T a, const T b, const T c) {
  const bool active = a != 0 && b >= 0 && c <= 0;
  const bool either = a == 0 || b > 0 || c < 0;
  return cpu::fixed::QuantizeSum<T>(active ? value : T{0},
                                    either ? T{0} : value, Bool<T>(active),
                                    Bool<T>(either));
}

template <typename T>
[[nodiscard]] bool RunMaskDsl(std::array<T, 8u> &value, std::array<T, 8u> &a,
                              std::array<T, 8u> &b, std::array<T, 8u> &c,
                              std::array<T, 8u> &out) {
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(rund::kernel::i64)) {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 63>()
          .template read<"value">(value.data())
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template write<"out">(out.data());
    } else {
      return rund::compute_dsl::bind(value.size())
          .template fixed<1, 31>()
          .template read<"value">(value.data())
          .template read<"a">(a.data())
          .template read<"b">(b.data())
          .template read<"c">(c.data())
          .template write<"out">(out.data());
    }
  }();
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-mask")
          .on(body)
          .map([](auto i, auto b) {
            auto value = b.template read<"value">();
            auto a = b.template read<"a">();
            auto bb = b.template read<"b">();
            auto c = b.template read<"c">();
            auto out = b.template write<"out">();
            const auto active =
                rund::compute_dsl::all(rund::compute_dsl::nonzero(a[i]),
                                       rund::compute_dsl::is_nonneg(bb[i]),
                                       rund::compute_dsl::is_nonpos(c[i]));
            const auto either =
                rund::compute_dsl::any(rund::compute_dsl::is_zero(a[i]),
                                       rund::compute_dsl::is_pos(bb[i]),
                                       rund::compute_dsl::is_neg(c[i]));
            out[i] = rund::compute_dsl::keep_if(value[i], active) +
                     rund::compute_dsl::zero_if(value[i], either) + active +
                     either;
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
[[nodiscard]] bool RunCase(std::array<T, 8u> &value, std::array<T, 8u> &a,
                           std::array<T, 8u> &b, std::array<T, 8u> &c) {
  std::array<T, 8u> out{};
  TEST_ASSERT(RunMaskDsl(value, a, b, c, out));
  for (std::size_t index = 0u; index < out.size(); ++index) {
    TEST_ASSERT(out[index] ==
                Expected(value[index], a[index], b[index], c[index]));
  }
  return true;
}

} // namespace

[[nodiscard]] bool RunsDeterministicMaskDslOps() {
  std::array<rund::kernel::i32, 8u> value32{3, -7, 11, -13, 17, -19, 23, -29};
  std::array<rund::kernel::i32, 8u> a32{1, 0, -2, 4, 0, 6, -8, 10};
  std::array<rund::kernel::i32, 8u> b32{0, -1, 2, 3, -4, 5, 0, -6};
  std::array<rund::kernel::i32, 8u> c32{0, 1, -2, 3, -4, 5, -6, 7};
  TEST_ASSERT(RunCase(value32, a32, b32, c32));

  std::array<rund::kernel::i64, 8u> value64{
      3, -7, 0x1000000000000000ll, -13, 17, -19, 23, -29};
  std::array<rund::kernel::i64, 8u> a64{1, 0, -2, 4, 0, 6, -8, 10};
  std::array<rund::kernel::i64, 8u> b64{0, -1, 2, 3, -4, 5, 0, -6};
  std::array<rund::kernel::i64, 8u> c64{0, 1, -2, 3, -4, 5, -6, 7};
  return RunCase(value64, a64, b64, c64);
}

} // namespace node_accel_contract
