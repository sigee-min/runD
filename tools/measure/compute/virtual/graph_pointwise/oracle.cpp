#include "internal.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::virtual_graph_pointwise {
namespace {

[[nodiscard]] bool add_exact(const std::uint64_t left,
                             const std::uint64_t right,
                             const std::uint64_t expected) noexcept {
  return left <= std::numeric_limits<std::uint64_t>::max() - right &&
         left + right == expected;
}

[[nodiscard]] bool
hash_cache_ok(const virtual_residency::ProductRouteEvidence &e) noexcept {
  return e.first_output_hash_observation_count == 0u &&
         e.last_output_hash_observation_count == 0u &&
         e.first_output_hash_reuse_count == 0u &&
         e.last_output_hash_reuse_count == 0u &&
         e.hash_observation_changes == 0u && e.hash_reuse_step_errors == 0u;
}

[[nodiscard]] bool route_ok(const virtual_residency::ProductRouteEvidence &e,
                            const Backend backend,
                            const std::uint64_t runs) noexcept {
  const std::uint64_t pages = Spec::PageCount * runs;
  const std::uint64_t output_bytes =
      Spec::ElementCount * sizeof(std::uint64_t) * runs;
  const std::uint64_t input_bytes = Spec::InputCount * output_bytes;
  const bool lifetime =
      (runs == 1u && e.cold_owner_runs == 1u && e.warm_reused_runs == 0u &&
       e.max_warm_rearm_count == 0u) ||
      (runs == SampleCount && e.cold_owner_runs == 0u &&
       e.warm_reused_runs == runs && e.max_warm_rearm_count >= runs);
  return e.backend == backend && e.owner_stable && e.production_slots &&
         e.owner_events == runs * 2u && e.prepare_attempts == runs &&
         e.prepared_runs == runs && e.executed_runs == runs &&
         e.successful_finals == runs && lifetime &&
         e.whole_run_preencoded_runs == 0u &&
         e.device_generated_recurrence_runs == runs &&
         e.fixed_native_storage_runs == runs &&
         e.fixed_common_storage_runs == runs &&
         e.gpu_addressable_backing_runs == runs &&
         e.public_gpu_addressable_backing_runs == 0u &&
         e.whole_run_staging_runs == runs &&
         e.bounded_external_page_service_runs == 0u &&
         e.one_native_submit_runs == runs &&
         e.host_service_turns_zero_runs == runs &&
         e.host_epoch_callbacks_zero_runs == runs &&
         e.aggregate_terminal_once_runs == runs &&
         e.bounded_page_io_runs == runs && e.coordinate_count == pages &&
         e.accepted_coordinates == pages &&
         e.gpu_completed_coordinates == pages && e.completed_prefix == pages &&
         e.forecasted_pages == pages && e.promoted_pages == pages &&
         e.drained_pages == pages && e.persisted_pages == pages &&
         e.overlap_reused_bytes == 0u &&
         e.gpu_backing_read_bytes == input_bytes &&
         e.gpu_backing_write_bytes == output_bytes &&
         e.native_submit_count == runs && e.epoch_native_submit_count == 0u &&
         e.payload_dispatch_count == runs && e.host_service_turn_count == 0u &&
         e.backend_epoch_callback_count == 0u &&
         e.host_epoch_callback_count == 0u && e.backing_wait_count == 0u &&
         e.backing_signal_count == 0u &&
         e.backing_acknowledgement_count == 0u &&
         e.final_callback_count == runs && e.queue_calls == runs &&
         e.public_handoff_count == runs && e.authority_accept_count == runs &&
         e.pipeline_terminal_count == Spec::StageCount * runs &&
         e.backing_publication_count == runs && e.completed_ns != 0u &&
         e.retained_bytes != 0u;
}

[[nodiscard]] bool phases_ok(const Result &result) noexcept {
  const auto &e = result.conditioning;
  const auto &cold = result.cold.route;
  const auto &warm = result.warm.route;
  return e.backend == result.backend && e.owner_stable &&
         e.owner_events == 2u && e.prepare_attempts == 1u &&
         e.prepared_runs == 1u && e.executed_runs == 1u &&
         e.successful_finals == 1u && hash_cache_ok(e) && hash_cache_ok(cold) &&
         hash_cache_ok(warm);
}

[[nodiscard]] bool transfer_ok(const ::rund::compute::Stats &stats,
                               const Backend backend,
                               const std::uint64_t input_bytes,
                               const std::uint64_t output_bytes) noexcept {
  const auto &submits = stats.transfer_submissions;
  const auto mapped = [](const std::uint64_t count, const std::uint64_t bytes,
                         const std::uint64_t expected) noexcept {
    return (count == 0u && bytes == 0u) || (count == 1u && bytes == expected);
  };
  if (submits.device_to_device != 0u) {
    return false;
  }
  if (backend == Backend::Metal) {
    return submits.host_to_device == 0u && submits.device_to_host == 0u &&
           stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u;
  }
  return backend == Backend::Vulkan &&
         mapped(submits.host_to_device, stats.uploaded_bytes, input_bytes) &&
         mapped(submits.device_to_host, stats.downloaded_bytes, output_bytes);
}

[[nodiscard]] bool stats_ok(const Facts &facts,
                            const Backend backend) noexcept {
  const auto &s = facts.stats;
  const auto &r = facts.residency;
  const std::uint64_t output_bytes = Spec::ElementCount * sizeof(std::uint64_t);
  const std::uint64_t input_bytes = Spec::InputCount * output_bytes;
  const std::uint64_t resource_count = Spec::InputCount + Spec::StageCount;
  const std::uint64_t frame_bytes = Spec::FrameElements * sizeof(std::uint64_t);
  const std::uint64_t logical_bytes = resource_count * output_bytes;
  const std::uint64_t page_bytes = resource_count * frame_bytes;
  return s.backend == backend && s.command_submits == 1u &&
         s.dispatches == 1u && s.final_dispatches == 1u &&
         transfer_ok(s, backend, input_bytes, output_bytes) &&
         s.internal_roundtrip_bytes == 0u && s.external_roundtrip_bytes == 0u &&
         r.logical_bytes == logical_bytes &&
         r.active_count == Spec::ElementCount && r.page_bytes == page_bytes &&
         r.page_count == Spec::PageCount &&
         r.frame_capacity == Spec::FrameCapacity &&
         r.epoch_count == Spec::BatchCount &&
         r.page_in_count == Spec::InputCount * Spec::PageCount &&
         r.page_out_count == Spec::PageCount &&
         r.page_in_bytes == input_bytes && r.page_out_bytes == output_bytes &&
         r.backing_read_bytes == input_bytes &&
         r.backing_write_bytes == output_bytes;
}

[[nodiscard]] bool proof_ok(const Facts &facts,
                            const Backend backend) noexcept {
  return facts.ok && facts.final_received && facts.owner_present &&
         facts.may_write && facts.graph_pointwise && facts.proof_valid &&
         facts.page_map_valid && !facts.quarantined &&
         facts.stats.backend == backend &&
         facts.input_count == Spec::InputCount &&
         facts.stage_count == Spec::StageCount &&
         facts.page_count == Spec::PageCount &&
         facts.tail_elements == Spec::TailElements &&
         facts.frame_capacity == Spec::FrameCapacity &&
         facts.batch_count == Spec::BatchCount &&
         facts.map_rows == Spec::MapRows && facts.workgroup_width == 256u &&
         facts.native_submits == 1u && facts.dispatches == 1u &&
         facts.finals == 1u && facts.epoch_submits == 0u &&
         facts.generated_pages == Spec::PageCount &&
         facts.completed_pages == Spec::PageCount &&
         facts.forecasted_pages == Spec::PageCount &&
         facts.promoted_pages == Spec::PageCount &&
         facts.drained_pages == Spec::PageCount &&
         facts.persisted_pages == Spec::PageCount && facts.host_turns == 0u &&
         facts.host_callbacks == 0u &&
         facts.gpu_read_bytes ==
             Spec::InputCount * Spec::ElementCount * sizeof(std::uint64_t) &&
         facts.gpu_write_bytes == Spec::ElementCount * sizeof(std::uint64_t) &&
         stats_ok(facts, backend);
}

[[nodiscard]] bool cpu_ok(const Facts &facts) noexcept {
  return facts.ok && facts.cpu_reference && facts.output_match &&
         facts.stats.backend == Backend::Cpu && !facts.owner_present &&
         facts.stats.command_submits == 0u &&
         facts.stats.dispatches == Spec::PageCount * Spec::StageCount;
}

enum class Leaf : std::uint32_t {
  DeviceValid,
  ColdFinite,
  ColdPositive,
  Timing,
  GraphHi,
  GraphLo,
  ColdGraphHi,
  ColdGraphLo,
  WarmGraphHi,
  WarmGraphLo,
  InputBefore,
  InputAfter,
  ExpectedHash,
  ColdOutputHash,
  WarmOutputHash,
  ColdOutput,
  WarmOutput,
  ColdVersion,
  WarmVersion,
  Owner,
  CpuCold,
  CpuWarm,
  CpuColdRoute,
  CpuWarmRoute,
  Backend,
  ColdProof,
  WarmProof,
  ColdRoute,
  WarmRoute,
  ColdHash,
  WarmHash,
  Phases,
  WarmAllocation,
  ProofDigest,
  ProofHi,
  ProofLo,
  Generation,
  Nonce,
  Count,
};

static_assert(static_cast<std::uint32_t>(Leaf::Count) <= 64u);

[[nodiscard]] const char *leaf_name(const Leaf leaf) noexcept {
  switch (leaf) {
  case Leaf::DeviceValid:
    return "device_valid";
  case Leaf::ColdFinite:
    return "cold_finite";
  case Leaf::ColdPositive:
    return "cold_positive";
  case Leaf::Timing:
    return "timing";
  case Leaf::GraphHi:
    return "graph_hi";
  case Leaf::GraphLo:
    return "graph_lo";
  case Leaf::ColdGraphHi:
    return "cold_graph_hi";
  case Leaf::ColdGraphLo:
    return "cold_graph_lo";
  case Leaf::WarmGraphHi:
    return "warm_graph_hi";
  case Leaf::WarmGraphLo:
    return "warm_graph_lo";
  case Leaf::InputBefore:
    return "input_before";
  case Leaf::InputAfter:
    return "input_after";
  case Leaf::ExpectedHash:
    return "expected_hash";
  case Leaf::ColdOutputHash:
    return "cold_output_hash";
  case Leaf::WarmOutputHash:
    return "warm_output_hash";
  case Leaf::ColdOutput:
    return "cold_output";
  case Leaf::WarmOutput:
    return "warm_output";
  case Leaf::ColdVersion:
    return "cold_version";
  case Leaf::WarmVersion:
    return "warm_version";
  case Leaf::Owner:
    return "owner_stable";
  case Leaf::CpuCold:
    return "cpu_cold";
  case Leaf::CpuWarm:
    return "cpu_warm";
  case Leaf::CpuColdRoute:
    return "cpu_cold_route";
  case Leaf::CpuWarmRoute:
    return "cpu_warm_route";
  case Leaf::Backend:
    return "backend";
  case Leaf::ColdProof:
    return "cold_proof";
  case Leaf::WarmProof:
    return "warm_proof";
  case Leaf::ColdRoute:
    return "cold_route";
  case Leaf::WarmRoute:
    return "warm_route";
  case Leaf::ColdHash:
    return "cold_hash";
  case Leaf::WarmHash:
    return "warm_hash";
  case Leaf::Phases:
    return "phases";
  case Leaf::WarmAllocation:
    return "warm_allocation";
  case Leaf::ProofDigest:
    return "proof_digest";
  case Leaf::ProofHi:
    return "proof_hi";
  case Leaf::ProofLo:
    return "proof_lo";
  case Leaf::Generation:
    return "generation";
  case Leaf::Nonce:
    return "nonce";
  case Leaf::Count:
    return "count";
  }
  return "unknown";
}

void mark(OracleDiagnostic &diagnostic, const Leaf leaf,
          const bool value) noexcept {
  if (value) {
    return;
  }
  const auto index = static_cast<std::uint32_t>(leaf);
  diagnostic.failed_mask |= std::uint64_t{1u} << index;
  if (diagnostic.first_leaf == std::numeric_limits<std::uint32_t>::max()) {
    diagnostic.first_leaf = index;
    diagnostic.first_leaf_name = leaf_name(leaf);
  }
}

} // namespace

OracleDiagnostic Diagnose(const Result &result) noexcept {
  OracleDiagnostic diagnostic{};
  const auto check = [&](const Leaf leaf, const bool value) noexcept {
    mark(diagnostic, leaf, value);
  };

  check(Leaf::DeviceValid, result.device_valid);
  check(Leaf::ColdFinite, std::isfinite(result.cold_us));
  check(Leaf::ColdPositive, result.cold_us > 0.0);
  check(Leaf::Timing,
        virtual_graph_residency::timing_summary_matches(result.timing));
  check(Leaf::GraphHi, result.graph_hi != 0u);
  check(Leaf::GraphLo, result.graph_lo != 0u);
  check(Leaf::ColdGraphHi, result.cold.graph_hi == result.graph_hi);
  check(Leaf::ColdGraphLo, result.cold.graph_lo == result.graph_lo);
  check(Leaf::WarmGraphHi, result.warm.graph_hi == result.graph_hi);
  check(Leaf::WarmGraphLo, result.warm.graph_lo == result.graph_lo);
  check(Leaf::InputBefore, result.input_digest_before == Spec::input_digest());
  check(Leaf::InputAfter,
        result.input_digest_after == result.input_digest_before);
  check(Leaf::ExpectedHash, result.expected_hash == Spec::expected_hash());
  check(Leaf::ColdOutputHash, result.cold.output_hash == result.expected_hash);
  check(Leaf::WarmOutputHash, result.warm.output_hash == result.expected_hash);
  check(Leaf::ColdOutput, result.cold.output_match);
  check(Leaf::WarmOutput, result.warm.output_match);
  check(Leaf::ColdVersion,
        add_exact(result.cold.version_before, 1u, result.cold.version_after));
  check(Leaf::WarmVersion, add_exact(result.warm.version_before, SampleCount,
                                     result.warm.version_after));
  check(Leaf::Owner, result.owner_stable);

  if (result.backend == Backend::Cpu) {
    check(Leaf::CpuCold, cpu_ok(result.cold));
    check(Leaf::CpuWarm, cpu_ok(result.warm));
    check(Leaf::CpuColdRoute,
          result.cold.route.backend == Backend::Unavailable);
    check(Leaf::CpuWarmRoute,
          result.warm.route.backend == Backend::Unavailable);
  } else if (result.backend == Backend::Metal ||
             result.backend == Backend::Vulkan) {
    check(Leaf::ColdProof, proof_ok(result.cold, result.backend));
    check(Leaf::WarmProof, proof_ok(result.warm, result.backend));
    check(Leaf::ColdRoute, route_ok(result.cold.route, result.backend, 1u));
    check(Leaf::WarmRoute,
          route_ok(result.warm.route, result.backend, SampleCount));
    check(Leaf::ColdHash, hash_cache_ok(result.cold.route));
    check(Leaf::WarmHash, hash_cache_ok(result.warm.route));
    check(Leaf::Phases, phases_ok(result));
    check(Leaf::WarmAllocation,
          result.warm.residency.samples_allocation_free(SampleCount));
    check(Leaf::ProofDigest,
          result.cold.proof_digest == result.warm.proof_digest);
    check(Leaf::ProofHi, result.cold.proof_hi == result.warm.proof_hi);
    check(Leaf::ProofLo, result.cold.proof_lo == result.warm.proof_lo);
    check(Leaf::Generation,
          result.cold.generation != 0u &&
              result.warm.generation > result.cold.generation);
    check(Leaf::Nonce, result.cold.nonce != 0u && result.warm.nonce != 0u &&
                           result.cold.nonce != result.warm.nonce);
  } else {
    check(Leaf::Backend, false);
  }
  return diagnostic;
}

[[nodiscard]] bool Validate(const Result &result) noexcept {
  return Diagnose(result).clean();
}

} // namespace rund::measure::compute::virtual_graph_pointwise
