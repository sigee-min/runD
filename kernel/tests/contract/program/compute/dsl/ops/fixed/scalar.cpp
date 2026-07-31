#include "contract/program/compute/dsl/ops/local.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_fixed_scalar_ops_build_deterministic_ir() {
  i32 lhs32[4]{};
  i32 rhs32[4]{};
  i32 out32[4]{};
  const auto body32 = rund::compute_dsl::bind(4u)
                          .fixed<1, 31>()
                          .param<"marker">(7)
                          .read<"lhs">(lhs32)
                          .read<"rhs">(rhs32)
                          .write<"out">(out32);
  const auto build32 = [&body32]() {
    return rund::compute_dsl::def("dsl-fixed-scalar-lane32")
        .on(body32)
        .map([](auto i, auto b) {
          auto marker = b.template param<"marker">();
          auto lhs = b.template read<"lhs">();
          auto rhs = b.template read<"rhs">();
          auto out = b.template write<"out">();

          auto comparison = rund::compute_dsl::predicate_and(
              rund::compute_dsl::ne(lhs[i], marker),
              rund::compute_dsl::predicate_or(
                  rund::compute_dsl::gt(lhs[i] + 1, rhs[i]),
                  rund::compute_dsl::ge(rhs[i], 0)));
          auto unary = rund::compute_dsl::neg(lhs[i]) + (-rhs[i]) +
                       rund::compute_dsl::abs(lhs[i]) +
                       rund::compute_dsl::abs_magnitude(rhs[i]) +
                       rund::compute_dsl::sign(rhs[i]);
          out[i] = rund::compute_dsl::select(
              rund::compute_dsl::predicate_not(comparison), unary, 0);
        });
  };

  const auto first32 = build32();
  const auto second32 = build32();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const rund::kernel::LoweringArtifact metal32 = rund::kernel::LowerComputeIR(
      first32.ir(), rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(metal32.source_text.find("].op=constant") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=neg") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=abs") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=abs_magnitude") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=sign") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=ne") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=gt") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=ge") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=predicate_not") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=predicate_and") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=predicate_or") !=
              std::string_view::npos);

  const rund::kernel::TilePhaseDescription phase{
      .phase_id = 93u,
      .tile_count = 4u,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .output_shards = 4u,
              .queue_slots = 4u,
              .task_slots = 4u,
          },
  };
  const rund::kernel::ComputeCaps caps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 4096u,
      .staging_bytes = 256u,
      .max_window_tiles = 4u,
      .subgroup_width = 32u,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::ComputeLimit limit{
      .staging_bytes = 256u,
      .max_window_tiles = 4u,
  };
  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(phase, first32.map(), caps, limit);
  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.op_hash_hi == first32.ir().op_hash_hi);
  TEST_ASSERT(plan.op_hash_lo == first32.ir().op_hash_lo);
  TEST_ASSERT(plan.bytes_per_tile == 16u);

  i64 lhs64[4]{};
  i64 rhs64[4]{};
  i64 out64[4]{};
  const auto body64 = rund::compute_dsl::bind(4u)
                          .fixed<1, 63>()
                          .param<"marker">(i64{7})
                          .read<"lhs">(lhs64)
                          .read<"rhs">(rhs64)
                          .write<"out">(out64);
  const auto fixed_lane64 = rund::compute_dsl::def("dsl-fixed-scalar-lane64")
                                .on(body64)
                                .map([](auto i, auto b) {
                                  auto marker = b.template param<"marker">();
                                  auto lhs = b.template read<"lhs">();
                                  auto rhs = b.template read<"rhs">();
                                  auto out = b.template write<"out">();

                                  out[i] = rund::compute_dsl::select(
                                      rund::compute_dsl::ne(lhs[i], marker),
                                      rund::compute_dsl::abs(-rhs[i]), i64{0});
                                });

  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(fixed_lane64.ir().scalar == rund::kernel::ComputeScalar::Lane64);
  return 0;
}

} // namespace

int RunComputeDslScalarOpsContract() {
  return test_compute_fixed_scalar_ops_build_deterministic_ir();
}

} // namespace program_compute_contract
