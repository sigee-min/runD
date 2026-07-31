#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildStandardizeOp() {
  T a[4]{};
  T b[4]{};
  T c[4]{};
  T d[4]{};
  T scale[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template read<"d">(d)
          .template read<"scale">(scale)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template read<"d">(d)
          .template read<"scale">(scale)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-standardize").on(body).map([](auto i, auto bnd) {
    auto a = bnd.template read<"a">();
    auto b = bnd.template read<"b">();
    auto c = bnd.template read<"c">();
    auto d = bnd.template read<"d">();
    auto scale = bnd.template read<"scale">();
    auto out = bnd.template write<"out">();
    out[i] =
        rund::compute_dsl::standardized(rund::compute_dsl::StandardizedOp::Cubic, a[i],
                                  b[i], scale[i]) +
        rund::compute_dsl::standardized(rund::compute_dsl::StandardizedOp::Quartic, a[i],
                                  b[i], scale[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Cubic, a[i], b[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Cubic, a[i], b[i],
                          c[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Cubic, a[i], b[i],
                          c[i], d[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Quartic, a[i], b[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Quartic, a[i], b[i],
                          c[i]) +
        rund::compute_dsl::mean(rund::compute_dsl::StandardizedOp::Quartic, a[i], b[i],
                          c[i], d[i]);
  });
}

[[nodiscard]] std::size_t CountOp(
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto &node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasStandardizeOps(const std::string &source) {
  return source.find("].op=sqrt") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos;
}

int test_compute_standardize_helpers_lower_through_existing_ops() {
  const auto first32 = BuildStandardizeOp<i32>();
  const auto second32 = BuildStandardizeOp<i32>();
  const auto fixed_lane64 = BuildStandardizeOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Eq) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasStandardizeOps(metal32.source_text));
  TEST_ASSERT(HasStandardizeOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsStandardizeContract() {
  return test_compute_standardize_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
