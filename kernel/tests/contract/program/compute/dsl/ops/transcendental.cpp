#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildTranscendentalOp() {
  T x[4]{};
  T y[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"x">(x)
          .template read<"y">(y)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"x">(x)
          .template read<"y">(y)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-transcendental").on(body).map(
      [](auto i, auto b) {
        auto x = b.template read<"x">();
        auto y = b.template read<"y">();
        auto out = b.template write<"out">();
        out[i] = rund::compute_dsl::sin(x[i]) + rund::compute_dsl::cos(x[i]) + rund::compute_dsl::tan(x[i]) +
                 rund::compute_dsl::exp(x[i]) + rund::compute_dsl::log(x[i]) + rund::compute_dsl::pow(x[i], y[i]);
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

[[nodiscard]] bool HasTranscendentalOps(const std::string& source) {
  return source.find("].op=sin") != std::string_view::npos &&
         source.find("].op=cos") != std::string_view::npos &&
         source.find("].op=tan") != std::string_view::npos &&
         source.find("].op=exp") != std::string_view::npos &&
         source.find("].op=log") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos;
}

int test_compute_transcendental_helpers_lower_as_fixed_ops() {
  const auto first32 = BuildTranscendentalOp<i32>();
  const auto second32 = BuildTranscendentalOp<i32>();
  const auto fixed_lane64 = BuildTranscendentalOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sin) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Cos) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Tan) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Exp) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Log) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasTranscendentalOps(metal32.source_text));
  TEST_ASSERT(HasTranscendentalOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsTranscendentalContract() {
  return test_compute_transcendental_helpers_lower_as_fixed_ops();
}

} // namespace program_compute_contract
