#include "contract/program/compute/dsl/ops/local.hpp"

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildNormOp() {
  T in_lo[4]{};
  T in_hi[4]{};
  T out_lo[4]{};
  T out_hi[4]{};
  T value[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<
              1, 63, rund::kernel::ComputeRounding::NearestEven,
              rund::kernel::ComputeOverflow::Saturate,
              rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"in_lo">(in_lo)
          .template read<"in_hi">(in_hi)
          .template read<"out_lo">(out_lo)
          .template read<"out_hi">(out_hi)
          .template read<"value">(value)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<
              1, 31, rund::kernel::ComputeRounding::NearestEven,
              rund::kernel::ComputeOverflow::Saturate,
              rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"in_lo">(in_lo)
          .template read<"in_hi">(in_hi)
          .template read<"out_lo">(out_lo)
          .template read<"out_hi">(out_hi)
          .template read<"value">(value)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-norm").on(body).map([](auto i, auto b) {
    auto in_lo = b.template read<"in_lo">();
    auto in_hi = b.template read<"in_hi">();
    auto out_lo = b.template read<"out_lo">();
    auto out_hi = b.template read<"out_hi">();
    auto value = b.template read<"value">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::remap(in_lo[i], in_hi[i], out_lo[i], out_hi[i],
                                value[i]) +
             rund::compute_dsl::unlerp(in_lo[i], in_hi[i], value[i]) +
             rund::compute_dsl::smoothstep(in_lo[i], in_hi[i], value[i]) +
             rund::compute_dsl::smootherstep(in_lo[i], in_hi[i], value[i]);
  });
}

[[nodiscard]] bool HasNormOps(const std::string &source) {
  return source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=le") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos;
}

int test_compute_norm_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildNormOp<i32>();
  const auto second32 = BuildNormOp<i32>();
  const auto fixed_lane64 = BuildNormOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasNormOps(metal32.source_text));
  TEST_ASSERT(HasNormOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsNormContract() {
  return test_compute_norm_helpers_build_deterministic_lowerable_ir();
}

} // namespace program_compute_contract
