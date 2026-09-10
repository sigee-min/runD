#include "test/assert.hpp"

#include <kernel/program/compute/graph/signature.hpp>
#include <kernel/program/compute/scatter/reduce.hpp>

#include <array>
#include <limits>
#include <string_view>
#include <vector>

namespace program_compute_contract {

template <typename T>
int CheckCpuScatterReduce(const rund::kernel::ComputeDomain domain,
                          const rund::kernel::ComputeFixedFormat format = {}) {
  using namespace rund::kernel;
  constexpr u64 capacity = 97u;
  std::array<T, capacity> values{};
  std::array<u32, capacity> indices{};
  std::array<u32, capacity> sorted{};
  for (u64 i = 0u; i < capacity; ++i) {
    values[i] = i % 3u == 0u ? std::numeric_limits<T>::max()
                             : (i % 3u == 1u ? T{10} : static_cast<T>(-20));
  }
  for (const auto op :
       {ScatterReduceOp::Sum, ScatterReduceOp::Min, ScatterReduceOp::Max}) {
    for (const u64 outputs : {1u, 31u, 32u, 33u, 4096u}) {
      const auto plan = PlanScatterReduce({.op = op,
                                           .domain = domain,
                                           .fixed_format = format,
                                           .element_count = capacity,
                                           .output_count = outputs});
      const auto cpu = PlanScatterReduceCpu(plan);
      TEST_ASSERT(plan.ok);
      TEST_ASSERT(cpu.scratch_words <= capacity);
      TEST_ASSERT((cpu.strategy == ScatterReduceCpuStrategy::SortedKeys) ==
                  (outputs == 4096u));
      std::vector<u32> scratch(cpu.scratch_words + 2u, 0xcdcdcdcdu);
      std::vector<T> actual(outputs, T{41});
      std::vector<T> expected(outputs, T{41});
      for (unsigned reuse = 0u; reuse < 2u; ++reuse) {
        for (u64 i = 0u; i < capacity; ++i) {
          indices[i] = reuse == 0u ? static_cast<u32>((i * 31u) % outputs)
                                   : static_cast<u32>(outputs - 1u);
        }
        for (const u64 count : {0u, 1u, 32u, 97u}) {
          const auto reference =
              scatter_reduce_reference_detail::ReferenceScatterReduce(
                  values.data(), indices.data(), expected.data(), count, plan,
                  domain == ComputeDomain::Fixed, sorted.data(), sorted.size());
          const auto result = ExecuteScatterReduceCpu(
              values.data(), indices.data(), actual.data(), count, plan, cpu,
              scratch.data() + 1u, cpu.scratch_words);
          TEST_ASSERT(reference.ok && result.ok);
          TEST_ASSERT(actual == expected);
          TEST_ASSERT(result.conflict_count == reference.conflict_count);
          TEST_ASSERT(result.first_rejected_ordinal == count);
          TEST_ASSERT(scratch.front() == 0xcdcdcdcdu);
          TEST_ASSERT(scratch.back() == 0xcdcdcdcdu);
        }
      }
      const auto protected_output = actual;
      indices[33u] = static_cast<u32>(outputs);
      indices[70u] = static_cast<u32>(outputs);
      const auto bad = ExecuteScatterReduceCpu(
          values.data(), indices.data(), actual.data(), capacity, plan, cpu,
          scratch.data() + 1u, cpu.scratch_words);
      TEST_ASSERT(!bad.ok && bad.first_rejected_ordinal == 33u);
      TEST_ASSERT(bad.conflict_count == 0u);
      TEST_ASSERT(actual == protected_output);
      const auto overflow = ExecuteScatterReduceCpu(
          values.data(), indices.data(), actual.data(), capacity + 1u, plan,
          cpu, scratch.data() + 1u, cpu.scratch_words);
      TEST_ASSERT(!overflow.ok && overflow.first_rejected_ordinal == capacity);
      TEST_ASSERT(actual == protected_output);
      const auto short_scratch = ExecuteScatterReduceCpu(
          values.data(), indices.data(), actual.data(), capacity, plan, cpu,
          scratch.data() + 1u, cpu.scratch_words - 1u);
      TEST_ASSERT(!short_scratch.ok);
      TEST_ASSERT(actual == protected_output);
      auto forged = cpu;
      ++forged.scratch_words;
      const auto invalid_plan = ExecuteScatterReduceCpu(
          values.data(), indices.data(), actual.data(), capacity, plan, forged,
          scratch.data(), scratch.size());
      TEST_ASSERT(!invalid_plan.ok);
      TEST_ASSERT(actual == protected_output);
      // Reuse the dirty scratch left by an invalid preflight.
      indices[33u] = 0u;
      indices[70u] = 0u;
      const auto reference =
          scatter_reduce_reference_detail::ReferenceScatterReduce(
              values.data(), indices.data(), expected.data(), capacity, plan,
              domain == ComputeDomain::Fixed, sorted.data(), sorted.size());
      const auto recovered = ExecuteScatterReduceCpu(
          values.data(), indices.data(), actual.data(), capacity, plan, cpu,
          scratch.data() + 1u, cpu.scratch_words);
      TEST_ASSERT(recovered.ok && reference.ok && actual == expected);
      TEST_ASSERT(recovered.conflict_count == reference.conflict_count);
    }
  }
  return 0;
}

int RunScatterReduceContract() {
  using namespace rund::kernel;
  constexpr ComputeFixedFormat fixed{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = ComputeRounding::NearestEven,
      .overflow = ComputeOverflow::Saturate,
      .approximation = ComputeApproximation::Deterministic,
  };
  TEST_ASSERT(CheckCpuScatterReduce<i32>(ComputeDomain::I32) == 0);
  TEST_ASSERT(CheckCpuScatterReduce<u32>(ComputeDomain::U32) == 0);
  TEST_ASSERT(CheckCpuScatterReduce<i64>(ComputeDomain::I64) == 0);
  TEST_ASSERT(CheckCpuScatterReduce<u64>(ComputeDomain::U64) == 0);
  TEST_ASSERT(CheckCpuScatterReduce<i32>(ComputeDomain::Fixed, fixed) == 0);
  auto fixed64 = fixed;
  fixed64.integer_bits = 32u;
  fixed64.fraction_bits = 32u;
  TEST_ASSERT(CheckCpuScatterReduce<i64>(ComputeDomain::Fixed, fixed64) == 0);
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
  auto large_desc = fixed_desc;
  large_desc.element_count = 1048576u;
  const auto large_plan = PlanScatterReduce(large_desc);
  TEST_ASSERT(large_plan.ok && large_plan.pass_count == 4u);
  TEST_ASSERT(large_plan.preflight.group_count == 256u);
  TEST_ASSERT(large_plan.status_bytes == 16u + 1024u);
  TEST_ASSERT(large_plan.temp_bytes ==
              large_plan.segment_bytes + large_plan.status_bytes + 24u);
  auto forged_large = large_plan;
  ++forged_large.preflight.partial_bytes;
  TEST_ASSERT(!ScatterReducePlanMatchesDesc(large_desc, forged_large));
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
