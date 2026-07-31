#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildMaskDsl() {
  T value[4]{};
  T a[4]{};
  T b[4]{};
  T c[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-mask").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto a = b.template read<"a">();
    auto bb = b.template read<"b">();
    auto c = b.template read<"c">();
    auto out = b.template write<"out">();

    const auto active = rund::compute_dsl::all(
        rund::compute_dsl::nonzero(a[i]), rund::compute_dsl::is_nonneg(bb[i]),
        rund::compute_dsl::is_nonpos(c[i]));
    const auto either = rund::compute_dsl::any(
        rund::compute_dsl::is_zero(a[i]), rund::compute_dsl::is_pos(bb[i]),
        rund::compute_dsl::is_neg(c[i]));
    out[i] = rund::compute_dsl::keep_if(value[i], active) +
             rund::compute_dsl::zero_if(value[i], either) + active + either;
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

[[nodiscard]] bool HasPredicateSelectLowering(const std::string& source) {
  return source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=ne") != std::string_view::npos &&
         source.find("].op=lt") != std::string_view::npos &&
         source.find("].op=le") != std::string_view::npos &&
         source.find("].op=gt") != std::string_view::npos &&
         source.find("].op=ge") != std::string_view::npos &&
         source.find("].op=predicate_and") != std::string_view::npos &&
         source.find("].op=predicate_or") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos;
}

int test_compute_mask_helpers_lower_through_predicate_ops() {
  const auto first32 = BuildMaskDsl<i32>();
  const auto second32 = BuildMaskDsl<i32>();
  const auto fixed_lane64 = BuildMaskDsl<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Eq) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Ne) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Lt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Le) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Gt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Ge) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
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
  TEST_ASSERT(HasPredicateSelectLowering(metal32.source_text));
  TEST_ASSERT(HasPredicateSelectLowering(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsMaskContract() {
  return test_compute_mask_helpers_lower_through_predicate_ops();
}

} // namespace program_compute_contract
