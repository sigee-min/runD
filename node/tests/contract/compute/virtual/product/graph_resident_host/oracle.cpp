#include "internal.hpp"

#include <cstdio>

namespace rund_node_test_virtual::product::graph_resident_host {
namespace {

[[nodiscard]] bool route_quiet(const Observation &observation) noexcept {
  return observation.route_kind == RouteKind::CpuRolling &&
         observation.owner_mask == 0u && observation.owner_count == 0u;
}

[[nodiscard]] bool owner_quiet(const rund::compute::Stats &stats) noexcept {
  const auto &residency = stats.pipeline.residency;
  return residency.window_handoff_count == 0u &&
         residency.window_batch_count == 0u &&
         residency.window_queue_call_count == 0u;
}

[[nodiscard]] bool facts_ok(const Observation &observation) noexcept {
  for (const BackingFacts &facts : observation.inputs) {
    if (facts.read_count != PageCount || facts.read_bytes != LogicalBytes) {
      return false;
    }
  }
  return observation.output_facts.write_count == PageCount &&
         observation.output_facts.write_bytes == LogicalBytes;
}

} // namespace

bool validate_case(const Case &test_case,
                   const Observation &observation) noexcept {
  const auto &stats = observation.after;
  const auto &residency = stats.pipeline.residency;
  const bool valid =
      observation.status && route_quiet(observation) &&
      observation.callbacks_quiet && test_case.state != nullptr &&
      test_case.state->geometry.route ==
          rund::compute::detail::VirtualRoute::GraphPointwise &&
      test_case.state->graph_input_resource_count == InputCount &&
      test_case.state->device_vsm_product_cache == nullptr &&
      test_case.pipeline.plan().residency.frame_capacity == 2u &&
      stats.backend == rund::compute::Backend::Cpu &&
      stats.command_submits == 0u && stats.uploaded_bytes == 0u &&
      stats.downloaded_bytes == 0u && stats.external_roundtrip_bytes == 0u &&
      residency.page_in_bytes == 0u && owner_quiet(stats) &&
      residency.epoch_count == StageCount * BatchCount &&
      residency.page_in_count == InputCount * PageCount &&
      residency.backing_read_bytes == InputCount * LogicalBytes &&
      residency.page_out_count == PageCount &&
      residency.backing_write_bytes == LogicalBytes && facts_ok(observation) &&
      observation.output_match &&
      observation.output_hash == Workload::expected_hash() &&
      observation.version_after == observation.version_before + 1u &&
      observation.receipts_idle && observation.quarantine_empty &&
      observation.distinct_backings;
  if (!valid) {
    std::fprintf(stderr,
                 "GraphResident Host valid=0 route=%u owners=0x%x/%u "
                 "epoch=%llu in=%llu read=%llu out=%llu write=%llu "
                 "submit=%llu output=%u receipts=%u quarantine=%u\n",
                 static_cast<unsigned>(observation.route_kind),
                 observation.owner_mask, observation.owner_count,
                 static_cast<unsigned long long>(residency.epoch_count),
                 static_cast<unsigned long long>(residency.page_in_count),
                 static_cast<unsigned long long>(residency.backing_read_bytes),
                 static_cast<unsigned long long>(residency.page_out_count),
                 static_cast<unsigned long long>(residency.backing_write_bytes),
                 static_cast<unsigned long long>(stats.command_submits),
                 observation.output_match ? 1u : 0u,
                 observation.receipts_idle ? 1u : 0u,
                 observation.quarantine_empty ? 1u : 0u);
    if (observation.failure.has()) {
      const auto &failure = observation.failure.first();
      std::fprintf(
          stderr,
          "GraphResident Host first epoch=%llu stage=%u batch=%llu phase=%u "
          "check=%u reason=%u close=%u cphase=%u ccheck=%u "
          "token=%llu generation=%llu frame=%u binding=%u claims=%u "
          "bkey=%llu/%llu/%llu/%llu/%llu/%llu/%u "
          "fkey=%llu/%llu/%llu/%llu/%llu/%llu/%u "
          "baccess=%u bflags=%u "
          "bnext=%llu bretain=%llu fstate=%u ftier=%u frole=%u "
          "fassigned=%u fnext=%llu fretain=%llu bdirty=%llu/%llu "
          "fdirty=%llu/%llu prior=%llu/%llu\n",
          static_cast<unsigned long long>(failure.epoch), failure.stage,
          static_cast<unsigned long long>(failure.batch),
          static_cast<unsigned>(failure.phase),
          static_cast<unsigned>(failure.check),
          static_cast<unsigned>(failure.reason), failure.has_close ? 1u : 0u,
          static_cast<unsigned>(failure.close_phase),
          static_cast<unsigned>(failure.close_check),
          static_cast<unsigned long long>(failure.close.token),
          static_cast<unsigned long long>(failure.close.generation),
          failure.close.frame_index, failure.close.binding_index,
          failure.close.claims,
          static_cast<unsigned long long>(failure.close.binding_key.backing),
          static_cast<unsigned long long>(failure.close.binding_key.version),
          static_cast<unsigned long long>(failure.close.binding_key.extent),
          static_cast<unsigned long long>(
              failure.close.binding_key.materialization_hi),
          static_cast<unsigned long long>(
              failure.close.binding_key.materialization_lo),
          static_cast<unsigned long long>(failure.close.binding_key.page),
          static_cast<unsigned>(failure.close.binding_key.domain),
          static_cast<unsigned long long>(failure.close.frame_key.backing),
          static_cast<unsigned long long>(failure.close.frame_key.version),
          static_cast<unsigned long long>(failure.close.frame_key.extent),
          static_cast<unsigned long long>(
              failure.close.frame_key.materialization_hi),
          static_cast<unsigned long long>(
              failure.close.frame_key.materialization_lo),
          static_cast<unsigned long long>(failure.close.frame_key.page),
          static_cast<unsigned>(failure.close.frame_key.domain),
          static_cast<unsigned>(failure.close.binding_access),
          static_cast<unsigned>(failure.close.flags),
          static_cast<unsigned long long>(failure.close.binding_next_use),
          static_cast<unsigned long long>(failure.close.binding_retain_until),
          static_cast<unsigned>(failure.close.frame_state),
          static_cast<unsigned>(failure.close.frame_tier),
          static_cast<unsigned>(failure.close.frame_role),
          failure.close.frame_assigned ? 1u : 0u,
          static_cast<unsigned long long>(failure.close.frame_next_use),
          static_cast<unsigned long long>(failure.close.frame_retain_until),
          static_cast<unsigned long long>(failure.close.binding_dirty.offset),
          static_cast<unsigned long long>(failure.close.binding_dirty.bytes),
          static_cast<unsigned long long>(failure.close.frame_dirty.offset),
          static_cast<unsigned long long>(failure.close.frame_dirty.bytes),
          static_cast<unsigned long long>(
              failure.close.binding_prior_dirty.offset),
          static_cast<unsigned long long>(
              failure.close.binding_prior_dirty.bytes));
    }
  }
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_resident_host
