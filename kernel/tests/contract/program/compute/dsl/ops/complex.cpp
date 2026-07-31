#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildComplexOp() {
  T ar[4]{};
  T ai[4]{};
  T br[4]{};
  T bi[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"ar">(ar)
          .template read<"ai">(ai)
          .template read<"br">(br)
          .template read<"bi">(bi)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"ar">(ar)
          .template read<"ai">(ai)
          .template read<"br">(br)
          .template read<"bi">(bi)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-complex").on(body).map([](auto i, auto b) {
    auto ar = b.template read<"ar">();
    auto ai = b.template read<"ai">();
    auto br = b.template read<"br">();
    auto bi = b.template read<"bi">();
    auto out = b.template write<"out">();
    out[i] =
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Conj, rund::compute_dsl::ComplexPart::Real, ar[i], ai[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Conj, rund::compute_dsl::ComplexPart::Imag, ar[i], ai[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Mul, rund::compute_dsl::ComplexPart::Real, ar[i], ai[i],
                   br[i], bi[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Mul, rund::compute_dsl::ComplexPart::Imag, ar[i], ai[i],
                   br[i], bi[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Abs, ar[i], ai[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Abs2, ar[i], ai[i]) +
        rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Phase, ar[i], ai[i]);
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

[[nodiscard]] bool HasComplexOps(const std::string& source) {
  return source.find("].op=neg") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sqrt") != std::string_view::npos &&
         source.find("].op=atan2") != std::string_view::npos;
}

int test_compute_complex_helpers_lower_as_scalar_pair_ops() {
  const auto first32 = BuildComplexOp<i32>();
  const auto second32 = BuildComplexOp<i32>();
  const auto fixed_lane64 = BuildComplexOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Neg) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Atan2) > 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasComplexOps(metal32.source_text));
  TEST_ASSERT(HasComplexOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsComplexContract() {
  return test_compute_complex_helpers_lower_as_scalar_pair_ops();
}

} // namespace program_compute_contract
