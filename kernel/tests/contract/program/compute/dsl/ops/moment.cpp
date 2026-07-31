#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildMomentOp() {
  T a[4]{};
  T b[4]{};
  T c[4]{};
  T d[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<
              1, 63, rund::kernel::ComputeRounding::NearestEven,
              rund::kernel::ComputeOverflow::Saturate,
              rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template read<"d">(d)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<
              1, 31, rund::kernel::ComputeRounding::NearestEven,
              rund::kernel::ComputeOverflow::Saturate,
              rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"a">(a)
          .template read<"b">(b)
          .template read<"c">(c)
          .template read<"d">(d)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-moment").on(body).map([](auto i, auto bind) {
    auto a = bind.template read<"a">();
    auto b = bind.template read<"b">();
    auto c = bind.template read<"c">();
    auto d = bind.template read<"d">();
    auto out = bind.template write<"out">();
    out[i] = rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Squared, a[i],
                                   b[i]) +
             rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Cubic, a[i],
                                   b[i]) +
             rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Quartic, a[i],
                                   b[i]) +
             rund::compute_dsl::var(a[i], b[i]) +
             rund::compute_dsl::var(a[i], b[i], c[i]) +
             rund::compute_dsl::var(a[i], b[i], c[i], d[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Cubic, a[i], b[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Cubic, a[i], b[i],
                               c[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Cubic, a[i], b[i],
                               c[i], d[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Quartic, a[i], b[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Quartic, a[i], b[i],
                               c[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::CenteredOp::Quartic, a[i], b[i],
                               c[i], d[i]) +
             rund::compute_dsl::rms(a[i], b[i]) +
             rund::compute_dsl::rms(a[i], b[i], c[i]) +
             rund::compute_dsl::rms(a[i], b[i], c[i], d[i]);
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

[[nodiscard]] bool HasMomentOps(const std::string& source) {
  return source.find("].op=constant") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=sqrt") != std::string_view::npos;
}

int test_compute_moment_helpers_lower_through_existing_ops() {
  const auto first32 = BuildMomentOp<i32>();
  const auto second32 = BuildMomentOp<i32>();
  const auto fixed_lane64 = BuildMomentOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Constant) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasMomentOps(metal32.source_text));
  TEST_ASSERT(HasMomentOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsMomentContract() {
  return test_compute_moment_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
