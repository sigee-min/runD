#include "test/assert.hpp"

#include <kernel/program/compute/graph/signature.hpp>
#include <kernel/program/compute/scatter/reduce.hpp>

#include <array>
#include <limits>
#include <string_view>

namespace program_compute_contract {

int RunScatterReduceContract() {
  using namespace rund::kernel;
  constexpr ComputeFixedFormat fixed{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Deterministic,
  };
  constexpr ScatterReduceDesc fixed_desc{
      .op = ScatterReduceOp::Sum,
      .domain = ComputeDomain::Fixed,
      .fixed_format = fixed,
      .element_count = 3u,
      .output_count = 2u,
      .count_source = ComputeCountSource::BufferU32,
  };
  constexpr ScatterReducePlan fixed_plan = PlanScatterReduce(fixed_desc);
  TEST_ASSERT(fixed_plan.ok);
  TEST_ASSERT(fixed_plan.element_bytes == 4u);
  TEST_ASSERT(fixed_plan.radix_pass_count == 0u);
  TEST_ASSERT(fixed_plan.fold_pass_count == 1u);
  TEST_ASSERT(fixed_plan.pass_count == 3u);
  TEST_ASSERT(fixed_plan.sorted_index_bytes == 0u);
  TEST_ASSERT(fixed_plan.sorted_value_bytes == 0u);
  TEST_ASSERT(fixed_plan.segment_bytes ==
              fixed_desc.output_count * sizeof(u32));
  TEST_ASSERT(fixed_plan.status_bytes == 4u * sizeof(u32));
  TEST_ASSERT(fixed_plan.indirect_bytes == 6u * sizeof(u32));
  TEST_ASSERT(fixed_plan.count_source == ComputeCountSource::BufferU32);
  TEST_ASSERT(ScatterReducePlanMatchesDesc(fixed_desc, fixed_plan));
  TEST_ASSERT(!ScatterReduceFoldParallel(fixed_plan));
  ScatterReducePlan forged = fixed_plan;
  ++forged.indirect_bytes;
  TEST_ASSERT(!ScatterReducePlanMatchesDesc(fixed_desc, forged));

  constexpr ScatterReducePlan unsupported_count = PlanScatterReduce({
      .op = ScatterReduceOp::Sum,
      .domain = ComputeDomain::U32,
      .element_count = static_cast<u64>(~u32{0u}) + 1u,
      .output_count = 1u,
  });
  TEST_ASSERT(!unsupported_count.ok);
  TEST_ASSERT(std::string_view{unsupported_count.reason} ==
              "compute_scatter_reduce_count_unsupported");

  const GraphSignature signature = GraphSignatureFor(fixed_plan);
  TEST_ASSERT(signature.ok);
  TEST_ASSERT(signature.kind == NodeKind::ScatterReduce);
  TEST_ASSERT(signature.value_count == 4u);
  TEST_ASSERT(signature.values[2u].kind == GraphValueKind::LogicalCount);
  TEST_ASSERT(signature.values[3u].role == BufferRole::Write);

  constexpr auto changed_op = HashScatterReduce(ScatterReduceDesc{
      .op = ScatterReduceOp::Max,
      .domain = fixed_desc.domain,
      .fixed_format = fixed_desc.fixed_format,
      .element_count = fixed_desc.element_count,
      .output_count = fixed_desc.output_count,
      .count_source = fixed_desc.count_source,
  });
  constexpr auto fixed_hash = HashScatterReduce(fixed_desc);
  TEST_ASSERT(changed_op.hi != fixed_hash.hi || changed_op.lo != fixed_hash.lo);

  constexpr ScatterReducePlan u32_plan = PlanScatterReduce({
      .op = ScatterReduceOp::Sum,
      .domain = ComputeDomain::U32,
      .element_count = 4u,
      .output_count = 3u,
  });
  TEST_ASSERT(ScatterReduceFoldParallel(u32_plan));
  const std::array<u32, 4u> values{3u, 7u, 11u, 13u};
  const std::array<u32, 4u> indices{1u, 0u, 1u, 1u};
  std::array<u32, 4u> sorted_indices{};
  std::array<u32, 3u> output{99u, 99u, 99u};
  const auto sum = ReferenceScatterReduceU32(
      values.data(), indices.data(), output.data(), values.size(), u32_plan,
      sorted_indices.data(), sorted_indices.size());
  TEST_ASSERT(sum.ok);
  TEST_ASSERT(sum.conflict_count == 2u);
  TEST_ASSERT(output[0u] == 7u);
  TEST_ASSERT(output[1u] == 27u);
  TEST_ASSERT(output[2u] == 0u);

  std::array<u32, 3u> missing_scratch_output{51u, 52u, 53u};
  const std::array<u32, 3u> missing_scratch_expected = missing_scratch_output;
  const auto missing_scratch = ReferenceScatterReduceU32(
      values.data(), indices.data(), missing_scratch_output.data(),
      values.size(), u32_plan, nullptr, 0u);
  TEST_ASSERT(!missing_scratch.ok);
  TEST_ASSERT(std::string_view{missing_scratch.reason} ==
              "compute_scatter_reduce_buffer_invalid");
  TEST_ASSERT(missing_scratch_output == missing_scratch_expected);

  // Invalid count and target are preflight failures: caller-visible output is
  // unchanged, rather than partially identity-cleared or folded.
  std::array<u32, 3u> protected_output{41u, 42u, 43u};
  const std::array<u32, 3u> protected_expected{41u, 42u, 43u};
  const auto oversized = ReferenceScatterReduceU32(
      values.data(), indices.data(), protected_output.data(), 5u, u32_plan,
      sorted_indices.data(), sorted_indices.size());
  TEST_ASSERT(!oversized.ok);
  TEST_ASSERT(std::string_view{oversized.reason} ==
              "compute_scatter_reduce_count_out_of_range");
  TEST_ASSERT(protected_output == protected_expected);
  const std::array<u32, 4u> bad_indices{1u, 0u, 3u, 1u};
  const auto invalid = ReferenceScatterReduceU32(
      values.data(), bad_indices.data(), protected_output.data(), 4u, u32_plan,
      sorted_indices.data(), sorted_indices.size());
  TEST_ASSERT(!invalid.ok);
  TEST_ASSERT(invalid.first_rejected_ordinal == 2u);
  TEST_ASSERT(protected_output == protected_expected);

  // Saturating addition is non-associative. The canonical answer is the
  // stable source-ordinal sequence, never a backend-dependent tree:
  // sat(sat(MAX + 10) - 20) == MAX - 20.
  const std::array<i32, 3u> fixed_values{std::numeric_limits<i32>::max(), 10,
                                         -20};
  const std::array<u32, 3u> fixed_indices{0u, 0u, 0u};
  std::array<i32, 2u> fixed_output{};
  const auto fixed_sum = ReferenceScatterReduceFixedI32(
      fixed_values.data(), fixed_indices.data(), fixed_output.data(), 3u,
      fixed_plan, sorted_indices.data(), sorted_indices.size());
  TEST_ASSERT(fixed_sum.ok);
  TEST_ASSERT(fixed_output[0u] == std::numeric_limits<i32>::max() - 20);
  TEST_ASSERT(fixed_output[1u] == 0);

  constexpr ScatterReducePlan min_plan = PlanScatterReduce({
      .op = ScatterReduceOp::Min,
      .domain = ComputeDomain::I32,
      .element_count = 3u,
      .output_count = 2u,
  });
  std::array<i32, 2u> min_output{};
  const auto minimum = ReferenceScatterReduceI32(
      fixed_values.data(), fixed_indices.data(), min_output.data(), 0u,
      min_plan, sorted_indices.data(), sorted_indices.size());
  TEST_ASSERT(minimum.ok);
  TEST_ASSERT(min_output[0u] == std::numeric_limits<i32>::max());
  TEST_ASSERT(min_output[1u] == std::numeric_limits<i32>::max());
  return 0;
}

} // namespace program_compute_contract
