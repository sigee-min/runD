#include "contract/program/compute/dsl/ops/local.hpp"

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildNoiseOp() {
  T cell[4]{};
  T t[4]{};
  T seed[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"cell">(cell)
          .template read<"t">(t)
          .template read<"seed">(seed)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"cell">(cell)
          .template read<"t">(t)
          .template read<"seed">(seed)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-noise").on(body).map([](auto i, auto b) {
    auto cell = b.template read<"cell">();
    auto t = b.template read<"t">();
    auto seed = b.template read<"seed">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::bit_xor(rund::compute_dsl::noise(cell[i], t[i], seed[i]),
                                  rund::compute_dsl::noise(cell[i], t[i]));
  });
}

[[nodiscard]] bool HasNoiseOps(const std::string &source) {
  return source.find("].op=add") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=bit_xor") != std::string_view::npos &&
         source.find("].op=shr_logical_const") != std::string_view::npos &&
         source.find("].op=bit_and") != std::string_view::npos;
}

int test_compute_noise_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildNoiseOp<i32>();
  const auto second32 = BuildNoiseOp<i32>();
  const auto fixed_lane64 = BuildNoiseOp<i64>();

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
  TEST_ASSERT(HasNoiseOps(metal32.source_text));
  TEST_ASSERT(HasNoiseOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsNoiseContract() {
  return test_compute_noise_helpers_build_deterministic_lowerable_ir();
}

} // namespace program_compute_contract
