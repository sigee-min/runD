#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildPolyOp() {
  T x[4]{};
  T c0[4]{};
  T c1[4]{};
  T c2[4]{};
  T c3[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"x">(x)
          .template read<"c0">(c0)
          .template read<"c1">(c1)
          .template read<"c2">(c2)
          .template read<"c3">(c3)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"x">(x)
          .template read<"c0">(c0)
          .template read<"c1">(c1)
          .template read<"c2">(c2)
          .template read<"c3">(c3)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-poly").on(body).map([](auto i, auto bind) {
    auto x = bind.template read<"x">();
    auto c0 = bind.template read<"c0">();
    auto c1 = bind.template read<"c1">();
    auto c2 = bind.template read<"c2">();
    auto c3 = bind.template read<"c3">();
    auto out = bind.template write<"out">();
    out[i] = rund::compute_dsl::poly(x[i], c0[i], c1[i], c2[i]) +
             rund::compute_dsl::poly(x[i], c0[i], c1[i], c2[i], c3[i]) +
             rund::compute_dsl::poly_deriv(x[i], c1[i], c2[i]) +
             rund::compute_dsl::poly_deriv(x[i], c1[i], c2[i], c3[i]);
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

[[nodiscard]] bool HasPolyOps(const std::string& source) {
  return source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos;
}

int test_compute_poly_helpers_lower_through_existing_ops() {
  const auto first32 = BuildPolyOp<i32>();
  const auto second32 = BuildPolyOp<i32>();
  const auto fixed_lane64 = BuildPolyOp<i64>();

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
  TEST_ASSERT(HasPolyOps(metal32.source_text));
  TEST_ASSERT(HasPolyOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsPolyContract() {
  return test_compute_poly_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
