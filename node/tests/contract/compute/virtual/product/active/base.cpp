#include "local.hpp"

#include <cstdio>
#include <cstring>

namespace rund_node_test_virtual::product::active {

[[nodiscard]] int CheckBase(ActiveFixture &fixture) {
  using namespace rund::compute;
  auto &input_backing = *fixture.input_backing;
  auto &output_backing = *fixture.output_backing;
  auto &prepared = *fixture.prepared;
  {
    auto partial_input_backing =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
    auto partial_output_backing =
        std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
    partial_output_backing->reset(TailPoison);
    auto partial_input =
        virtual_buffer<std::int32_t>(LogicalElements, partial_input_backing);
    auto partial_output =
        virtual_buffer<std::int32_t>(LogicalElements, partial_output_backing);
    auto cold_partial =
        partial_input && partial_output &&
                partial_input_backing->seed(
                    std::as_bytes(std::span{fixture.seeded}))
            ? virtual_pipeline(*fixture.program, *partial_input,
                               *partial_output, ResidencyConfig{})
            : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                  Reason::PipelineInvalid);
    constexpr std::size_t ColdPartialCount = 7u;
    std::array<std::byte, LogicalBytes> partial_observed{};
    if (!cold_partial || !cold_partial->run(ColdPartialCount) ||
        !partial_output_backing->observe(partial_observed) ||
        !PrefixAndTail(partial_observed, ColdPartialCount)) {
      return 5;
    }
    const Stats partial_stats = cold_partial->stats();
    if (partial_stats.pipeline.verified_step_count != 1u ||
        partial_stats.dispatches != 1u ||
        partial_stats.command_submits !=
            (fixture.backend == Backend::Cpu ? 0u : 1u) ||
        partial_stats.pipeline.residency.epoch_count != 1u ||
        partial_stats.pipeline.residency.page_in_count != 1u ||
        partial_stats.pipeline.residency.page_out_count != 1u) {
      return 5;
    }
  }
  fixture.plan = prepared.plan();
  fixture.memory = prepared.memory();
  fixture.input_identity = input_backing.identity();
  fixture.output_identity = output_backing.identity();
  const BackingFacts overflow_input = input_backing.facts();
  const BackingFacts overflow_output = output_backing.facts();
  if (prepared.run(LogicalElements + 1u).reason() != Reason::ShapeMismatch ||
      input_backing.facts().read_count != overflow_input.read_count ||
      output_backing.facts().write_count != overflow_output.write_count) {
    return 5;
  }

  std::array<std::byte, LogicalBytes> cold_observed{};
  if (!prepared.run() || !output_backing.observe(cold_observed)) {
    return 6;
  }
  const Stats cold_stats = prepared.stats();
  const auto cold_profile = prepared.profile();
  const bool cold_profile_present = static_cast<bool>(cold_profile);
  const bool cold_content_exact = PrefixAndTail(cold_observed, LogicalElements);
  const bool cold_active_count =
      cold_stats.pipeline.residency.active_count == LogicalElements;
  const std::uint64_t cold_golden_hash = HashValues(fixture.golden);
  const std::uint64_t cold_profile_hash =
      cold_profile_present ? cold_profile->execution().output_hash : 0u;
  const bool cold_stats_hash = cold_stats.output_hash == cold_golden_hash;
  const bool cold_profile_hash_matches =
      cold_profile_hash == cold_stats.output_hash;
  const bool cold_plan_same = prepared.plan() == fixture.plan;
  const bool cold_input_identity_same =
      input_backing.identity() == fixture.input_identity;
  const bool cold_output_identity_same =
      output_backing.identity() == fixture.output_identity;
  if (!cold_profile_present || !cold_content_exact || !cold_active_count ||
      !cold_stats_hash || !cold_profile_hash_matches || !cold_plan_same ||
      !cold_input_identity_same || !cold_output_identity_same) {
    std::size_t mismatch_index = LogicalElements;
    std::int32_t mismatch_actual = 0;
    std::int32_t mismatch_expected = 0;
    for (std::size_t index = 0u; index < LogicalElements; ++index) {
      std::memcpy(&mismatch_actual,
                  cold_observed.data() + index * sizeof(mismatch_actual),
                  sizeof(mismatch_actual));
      mismatch_expected = fixture.golden[index];
      if (mismatch_actual != mismatch_expected) {
        mismatch_index = index;
        break;
      }
    }
    const bool cold_tail_valid = output_backing.tail_poisoned();
    if (mismatch_index != LogicalElements) {
      std::uint32_t actual_bits = 0u;
      std::uint32_t expected_bits = 0u;
      std::memcpy(&actual_bits, &mismatch_actual, sizeof(actual_bits));
      std::memcpy(&expected_bits, &mismatch_expected, sizeof(expected_bits));
      std::fprintf(stderr,
                   "virtual active cold_output_first_mismatch index=%zu "
                   "actual=%d expected=%d actual_bytes=%02x%02x%02x%02x "
                   "expected_bytes=%02x%02x%02x%02x\n",
                   mismatch_index, mismatch_actual, mismatch_expected,
                   static_cast<unsigned>(actual_bits & 0xffu),
                   static_cast<unsigned>((actual_bits >> 8u) & 0xffu),
                   static_cast<unsigned>((actual_bits >> 16u) & 0xffu),
                   static_cast<unsigned>((actual_bits >> 24u) & 0xffu),
                   static_cast<unsigned>(expected_bits & 0xffu),
                   static_cast<unsigned>((expected_bits >> 8u) & 0xffu),
                   static_cast<unsigned>((expected_bits >> 16u) & 0xffu),
                   static_cast<unsigned>((expected_bits >> 24u) & 0xffu));
    } else {
      std::fprintf(stderr,
                   "virtual active cold_output_first_mismatch index=none\n");
    }
    std::fprintf(
        stderr,
        "virtual active cold_evidence profile_present=%u profile_reason=%u "
        "content=%u tail_valid=%u active_count=%llu expected_active=%llu "
        "stats_hash=%llx golden_hash=%llx profile_hash=%llx "
        "plan_same=%u input_identity=%p saved_input_identity=%p "
        "output_identity=%p saved_output_identity=%p\n",
        static_cast<unsigned>(cold_profile_present),
        static_cast<unsigned>(cold_profile.reason()),
        static_cast<unsigned>(cold_content_exact),
        static_cast<unsigned>(cold_tail_valid),
        static_cast<unsigned long long>(
            cold_stats.pipeline.residency.active_count),
        static_cast<unsigned long long>(LogicalElements),
        static_cast<unsigned long long>(cold_stats.output_hash),
        static_cast<unsigned long long>(cold_golden_hash),
        static_cast<unsigned long long>(cold_profile_hash),
        static_cast<unsigned>(cold_plan_same), input_backing.identity(),
        fixture.input_identity, output_backing.identity(),
        fixture.output_identity);
    return 7;
  }
  output_backing.reset(TailPoison);
  return 0;
}

} // namespace rund_node_test_virtual::product::active
