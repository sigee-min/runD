#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildCorrOp() {
  T x0[4]{};
  T x1[4]{};
  T x2[4]{};
  T x3[4]{};
  T y0[4]{};
  T y1[4]{};
  T y2[4]{};
  T y3[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"x0">(x0)
          .template read<"x1">(x1)
          .template read<"x2">(x2)
          .template read<"x3">(x3)
          .template read<"y0">(y0)
          .template read<"y1">(y1)
          .template read<"y2">(y2)
          .template read<"y3">(y3)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"x0">(x0)
          .template read<"x1">(x1)
          .template read<"x2">(x2)
          .template read<"x3">(x3)
          .template read<"y0">(y0)
          .template read<"y1">(y1)
          .template read<"y2">(y2)
          .template read<"y3">(y3)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-corr").on(body).map([](auto i, auto bind) {
    auto x0 = bind.template read<"x0">();
    auto x1 = bind.template read<"x1">();
    auto x2 = bind.template read<"x2">();
    auto x3 = bind.template read<"x3">();
    auto y0 = bind.template read<"y0">();
    auto y1 = bind.template read<"y1">();
    auto y2 = bind.template read<"y2">();
    auto y3 = bind.template read<"y3">();
    auto out = bind.template write<"out">();
    out[i] = rund::compute_dsl::cov(x0[i], x1[i], y0[i], y1[i]) +
             rund::compute_dsl::cov(x0[i], x1[i], x2[i], y0[i], y1[i], y2[i]) +
             rund::compute_dsl::cov(x0[i], x1[i], x2[i], x3[i], y0[i], y1[i],
                                y2[i], y3[i]) +
             rund::compute_dsl::corr(x0[i], x1[i], y0[i], y1[i]) +
             rund::compute_dsl::corr(x0[i], x1[i], x2[i], y0[i], y1[i], y2[i]) +
             rund::compute_dsl::corr(x0[i], x1[i], x2[i], x3[i], y0[i], y1[i],
                                 y2[i], y3[i]);
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

[[nodiscard]] bool HasCorrOps(const std::string& source) {
  return source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=sqrt") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos;
}

int test_compute_corr_helpers_lower_through_existing_ops() {
  const auto first32 = BuildCorrOp<i32>();
  const auto second32 = BuildCorrOp<i32>();
  const auto fixed_lane64 = BuildCorrOp<i64>();

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
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasCorrOps(metal32.source_text));
  TEST_ASSERT(HasCorrOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsCorrContract() {
  return test_compute_corr_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
