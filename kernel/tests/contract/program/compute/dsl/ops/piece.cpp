#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildPieceOp() {
  T value[4]{};
  T limit[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"limit">(limit)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"limit">(limit)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-piece").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto limit = b.template read<"limit">();
    auto out = b.template write<"out">();
    out[i] = rund::compute_dsl::clip(value[i], limit[i]) +
             rund::compute_dsl::huber(value[i], limit[i]) +
             rund::compute_dsl::positive_part(value[i]) +
             rund::compute_dsl::negative_part(value[i]);
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

[[nodiscard]] bool HasPieceOps(const std::string& source) {
  return source.find("].op=abs") != std::string_view::npos &&
         source.find("].op=neg_positive_fixed") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=le") != std::string_view::npos &&
         source.find("].op=min") != std::string_view::npos &&
         source.find("].op=max") != std::string_view::npos;
}

int test_compute_piece_helpers_lower_through_existing_ops() {
  const auto first32 = BuildPieceOp<i32>();
  const auto second32 = BuildPieceOp<i32>();
  const auto fixed_lane64 = BuildPieceOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Abs) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::NegPositiveFixed) >
              0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Clamp) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Le) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Min) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Max) > 0u);
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
  TEST_ASSERT(HasPieceOps(metal32.source_text));
  TEST_ASSERT(HasPieceOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsPieceContract() {
  return test_compute_piece_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
