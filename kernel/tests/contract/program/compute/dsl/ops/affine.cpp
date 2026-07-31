#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildAffineOp() {
  T m00[4]{};
  T m01[4]{};
  T m02[4]{};
  T m10[4]{};
  T m11[4]{};
  T m12[4]{};
  T m20[4]{};
  T m21[4]{};
  T m22[4]{};
  T tx[4]{};
  T ty[4]{};
  T tz[4]{};
  T x[4]{};
  T y[4]{};
  T z[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"m00">(m00)
          .template read<"m01">(m01)
          .template read<"m02">(m02)
          .template read<"m10">(m10)
          .template read<"m11">(m11)
          .template read<"m12">(m12)
          .template read<"m20">(m20)
          .template read<"m21">(m21)
          .template read<"m22">(m22)
          .template read<"tx">(tx)
          .template read<"ty">(ty)
          .template read<"tz">(tz)
          .template read<"x">(x)
          .template read<"y">(y)
          .template read<"z">(z)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"m00">(m00)
          .template read<"m01">(m01)
          .template read<"m02">(m02)
          .template read<"m10">(m10)
          .template read<"m11">(m11)
          .template read<"m12">(m12)
          .template read<"m20">(m20)
          .template read<"m21">(m21)
          .template read<"m22">(m22)
          .template read<"tx">(tx)
          .template read<"ty">(ty)
          .template read<"tz">(tz)
          .template read<"x">(x)
          .template read<"y">(y)
          .template read<"z">(z)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-affine").on(body).map([](auto i, auto bind) {
    auto m00 = bind.template read<"m00">();
    auto m01 = bind.template read<"m01">();
    auto m02 = bind.template read<"m02">();
    auto m10 = bind.template read<"m10">();
    auto m11 = bind.template read<"m11">();
    auto m12 = bind.template read<"m12">();
    auto m20 = bind.template read<"m20">();
    auto m21 = bind.template read<"m21">();
    auto m22 = bind.template read<"m22">();
    auto tx = bind.template read<"tx">();
    auto ty = bind.template read<"ty">();
    auto tz = bind.template read<"tz">();
    auto x = bind.template read<"x">();
    auto y = bind.template read<"y">();
    auto z = bind.template read<"z">();
    auto out = bind.template write<"out">();
    out[i] = rund::compute_dsl::aff(rund::compute_dsl::Axis::X, m00[i], m01[i], tx[i],
                              x[i], y[i]) +
             rund::compute_dsl::aff(rund::compute_dsl::Axis::Y, m10[i], m11[i], ty[i],
                              x[i], y[i]) +
             rund::compute_dsl::aff(rund::compute_dsl::Axis::X, m00[i], m01[i], m02[i],
                              tx[i], x[i], y[i], z[i]) +
             rund::compute_dsl::aff(rund::compute_dsl::Axis::Y, m10[i], m11[i], m12[i],
                              ty[i], x[i], y[i], z[i]) +
             rund::compute_dsl::aff(rund::compute_dsl::Axis::Z, m20[i], m21[i], m22[i],
                              tz[i], x[i], y[i], z[i]);
  });
}

[[nodiscard]] std::size_t CountOp(
    const rund::kernel::compute_lowering_detail::ParsedIR& parsed,
    const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto& node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasAffineOps(const std::string& source) {
  return source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos;
}

int test_compute_affine_helpers_lower_through_existing_ops() {
  const auto first32 = BuildAffineOp<i32>();
  const auto second32 = BuildAffineOp<i32>();
  const auto fixed_lane64 = BuildAffineOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasAffineOps(metal32.source_text));
  TEST_ASSERT(HasAffineOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsAffineContract() {
  return test_compute_affine_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
