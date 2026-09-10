#include "internal.hpp"
#include "oracle_detail.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_resident {

namespace {

[[nodiscard]] bool run_ok(const Observation &observation,
                          const std::size_t index,
                          const std::uint64_t expected_hash) noexcept {
  const std::uint32_t total_classes =
      observation.internal_owners + static_cast<std::uint32_t>(InputCount) + 1u;
  return oracle_detail::route_ok(observation) &&
         oracle_detail::native_ok(observation, PageCount, ElementCount) &&
         total_classes == TotalClassCount && total_classes == 7u &&
         observation.alias_reuse &&
         oracle_detail::output_ok(observation, expected_hash) &&
         observation.version_after == observation.version_before + 1u &&
         oracle_detail::stats_ok(observation, PageCount, ElementCount) &&
         oracle_detail::time_ok(observation) &&
         oracle_detail::completion_ok(observation) && index < RunCount;
}

[[nodiscard]] bool u32_run_ok(const Observation &observation,
                              const std::size_t index,
                              const std::uint64_t expected_hash) noexcept {
  return oracle_detail::route_ok(observation) &&
         observation.proof_scalar == static_cast<std::uint8_t>(
                                         rund::kernel::ComputeScalar::Lane32) &&
         observation.proof_domain == static_cast<std::uint8_t>(
                                         rund::kernel::ComputeDomain::U32) &&
         observation.proof_element_bytes == sizeof(std::uint32_t) &&
         oracle_detail::native_ok(observation, PageCount, ElementCount,
                                  sizeof(std::uint32_t)) &&
         observation.internal_owners == InternalOwnerCount &&
         observation.alias_reuse &&
         oracle_detail::output_ok(observation, expected_hash) &&
         observation.version_after == observation.version_before + 1u &&
         oracle_detail::stats_ok(observation, PageCount, ElementCount,
                                 sizeof(std::uint32_t)) &&
         oracle_detail::time_ok(observation) &&
         oracle_detail::completion_ok(observation) && index < RunCount;
}

[[nodiscard]] bool
dynamic_ok(const Case &test_case, const rund::compute::Backend backend,
           const Observation &observation) noexcept {
  constexpr std::uint64_t DynamicStages = StageCount;
  constexpr std::uint64_t DynamicPageCount = 6u;
  constexpr std::uint64_t DynamicElementCount =
      DynamicPageCount * FrameElements - TailElements;
  const std::shared_ptr<oracle_detail::Owner> retained =
      oracle_detail::owner_of(test_case);
  const std::uint64_t expected_hash = Workload::hash(
      std::span<const std::uint64_t>{test_case.expected});
  const bool owner_ok =
      retained != nullptr && retained->proof != nullptr &&
      retained->evidence != nullptr &&
      retained->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphResident &&
      retained->proof->graph_resident.stage_count == DynamicStages &&
      retained->proof->graph_resident.resource_count == 8u &&
      retained->proof->graph_wavefront.stage_count == DynamicStages &&
      retained->proof->graph_wavefront.batch_count == BatchCount &&
      retained->cold_prepare_count == 1u && retained->warm_rearm_count == 0u &&
      retained->evidence->public_handoff_count == 1u &&
      retained->evidence->authority_accept_count == 1u &&
      retained->evidence->pipeline_terminal_count == DynamicStages &&
      retained->evidence->backing_publication_count == 1u &&
      retained->evidence->final_received && !retained->evidence->quarantined;
  const bool valid =
      (backend == rund::compute::Backend::Metal ||
       backend == rund::compute::Backend::Vulkan) &&
      oracle_detail::route_ok(observation) &&
      oracle_detail::native_ok(observation, DynamicPageCount,
                               DynamicElementCount) &&
      oracle_detail::output_ok(observation, expected_hash) &&
      observation.version_after == observation.version_before + 1u &&
      oracle_detail::stats_ok(observation, DynamicPageCount,
                              DynamicElementCount) &&
      oracle_detail::time_ok(observation) &&
      oracle_detail::completion_ok(observation);
  if (!valid || !owner_ok) {
    std::fprintf(stderr,
                 "GraphResident Q6 valid=%u owner=%u submit=%llu dispatch=%llu "
                 "final=%llu stages=%llu resources=%u steps=%llu pages=%llu/%llu "
                 "host=%u version=%llu->%llu output=%u\n",
                 static_cast<unsigned>(valid), static_cast<unsigned>(owner_ok),
                 static_cast<unsigned long long>(observation.native_submit_count),
                 static_cast<unsigned long long>(observation.dispatch_count),
                 static_cast<unsigned long long>(observation.final_count),
                 static_cast<unsigned long long>(
                     observation.wavefront_steps / BatchCount),
                 observation.resource_count,
                 static_cast<unsigned long long>(observation.wavefront_steps),
                 static_cast<unsigned long long>(observation.generated_pages),
                 static_cast<unsigned long long>(observation.completed_pages),
                 static_cast<unsigned>(observation.host_service),
                 static_cast<unsigned long long>(observation.version_before),
                 static_cast<unsigned long long>(observation.version_after),
                 static_cast<unsigned>(observation.output_match));
    return false;
  }
  std::fprintf(stdout,
               "GraphResident Q6 positive native={submit=%llu dispatch=%llu "
               "stages=%llu steps=%llu pages=%llu/%llu/%llu/%llu/%llu host=0 "
               "final=%llu publication=1 version=+1} output=1\n",
               static_cast<unsigned long long>(observation.native_submit_count),
               static_cast<unsigned long long>(observation.dispatch_count),
               static_cast<unsigned long long>(DynamicStages),
               static_cast<unsigned long long>(observation.wavefront_steps),
               static_cast<unsigned long long>(observation.generated_pages),
               static_cast<unsigned long long>(observation.completed_pages),
               static_cast<unsigned long long>(observation.forecasted_pages),
               static_cast<unsigned long long>(observation.promoted_pages),
               static_cast<unsigned long long>(observation.persisted_pages),
               static_cast<unsigned long long>(observation.final_count));
  return true;
}

void print_diagnostics(const std::span<const Observation> observations,
                       const std::uint64_t expected_hash, const bool valid,
                       const std::shared_ptr<oracle_detail::Owner> &retained,
                       const bool owner_valid) noexcept {
  bool shape_ok = !observations.empty();
  bool alias_ok = shape_ok;
  bool internal_ok = shape_ok;
  bool class_ok = shape_ok;
  bool output_ok = shape_ok;
  for (const Observation &observation : observations) {
    const std::uint32_t total_classes = observation.internal_owners +
                                        static_cast<std::uint32_t>(InputCount) +
                                        1u;
    shape_ok = shape_ok && observation.graph_resident &&
               observation.resource_count == 8u;
    alias_ok = alias_ok && observation.alias_reuse;
    internal_ok =
        internal_ok && observation.internal_owners == InternalOwnerCount;
    class_ok =
        class_ok && total_classes == TotalClassCount && total_classes == 7u;
    output_ok = output_ok && observation.output_match &&
                observation.output_hash == expected_hash;
  }
  const oracle_detail::Continuity sequence =
      oracle_detail::continuity(observations);
  const bool staged_output =
      !observations.empty() && observations.front().staged_output;
  std::fprintf(
      stderr,
      "GraphResident diag staged=%u valid=%u owner=%u groups{shape=%u alias=%u "
      "internal=%u classes=%u output=%u "
      "continuity{version=%u proof=%u type=%u generation=%u nonce=%u all=%u}}\n",
      static_cast<unsigned>(staged_output), static_cast<unsigned>(valid),
      static_cast<unsigned>(owner_valid), static_cast<unsigned>(shape_ok),
      static_cast<unsigned>(alias_ok), static_cast<unsigned>(internal_ok),
      static_cast<unsigned>(class_ok), static_cast<unsigned>(output_ok),
      static_cast<unsigned>(sequence.version),
      static_cast<unsigned>(sequence.proof),
      static_cast<unsigned>(sequence.type),
      static_cast<unsigned>(sequence.generation),
      static_cast<unsigned>(sequence.nonce),
      static_cast<unsigned>(sequence.all()));

  if (retained == nullptr || retained->proof == nullptr ||
      retained->evidence == nullptr) {
    std::fprintf(stderr, "GraphResident owner{present=0}\n");
  } else {
    const auto &proof = *retained->proof;
    const auto &graph = proof.graph_resident;
    const auto &wavefront = proof.graph_wavefront;
    const auto &evidence = *retained->evidence;
    const auto &native = evidence.native;
    const std::uint32_t endpoints = static_cast<std::uint32_t>(InputCount + 1u);
    const std::uint32_t total_classes = graph.owner_binding_count + endpoints;
    std::fprintf(
        stderr,
        "GraphResident owner{present=1 topo=%u frame=%u batch=%u "
        "resource=%u internal=%u endpoints=%u total=%u cold=%llu warm=%llu "
        "handoff=%llu "
        "accept=%llu terminal=%llu publication=%llu final=%u quarantine=%u "
        "host_turn=%llu host_cb=%llu}\n",
        static_cast<unsigned>(proof.topology), wavefront.frame_capacity,
        wavefront.batch_count, graph.resource_count, graph.owner_binding_count,
        endpoints, total_classes,
        static_cast<unsigned long long>(retained->cold_prepare_count),
        static_cast<unsigned long long>(retained->warm_rearm_count),
        static_cast<unsigned long long>(evidence.public_handoff_count),
        static_cast<unsigned long long>(evidence.authority_accept_count),
        static_cast<unsigned long long>(evidence.pipeline_terminal_count),
        static_cast<unsigned long long>(evidence.backing_publication_count),
        static_cast<unsigned>(evidence.final_received),
        static_cast<unsigned>(evidence.quarantined),
        static_cast<unsigned long long>(native.host_service_turn_count),
        static_cast<unsigned long long>(native.host_epoch_callback_count));
  }

  for (std::size_t index = 0u; index < observations.size(); ++index) {
    const Observation &observation = observations[index];
    const auto &after = observation.after;
    const auto &after_residency = after.pipeline.residency;
    std::fprintf(
        stderr,
        "GraphResident run=%zu run_ok=%u status=%u route=%u owner=%u/%u "
        "v=%llu->%llu "
        "proof_digest=%llu native_proof=%llu:%llu frame=%llu gen=%llu "
        "nonce=%llu "
        "native{sub=%llu epoch=%llu dispatch=%llu tile=%llu final=%llu "
        "pages=%llu/%llu/%llu/%llu/%llu wave=%llu gpu=%llu/%llu "
        "time=%llu/%llu/%llu/%llu host=%u quarantine=%u} "
        "stats{sub=%llu dispatch=%llu final=%llu epochs=%llu "
        "page_in=%llu/%llu page_out=%llu/%llu "
        "backing=%llu/%llu xfer=%llu/%llu/%llu} "
        "out=%u hash=%llu/%llu\n",
        index, static_cast<unsigned>(run_ok(observation, index, expected_hash)),
        observation.status_reason,
        static_cast<unsigned>(observation.route_kind),
        observation.accepted_owner_mask, observation.accepted_owner_count,
        static_cast<unsigned long long>(observation.version_before),
        static_cast<unsigned long long>(observation.version_after),
        static_cast<unsigned long long>(observation.proof_digest),
        static_cast<unsigned long long>(observation.native_proof_hi),
        static_cast<unsigned long long>(observation.native_proof_lo),
        static_cast<unsigned long long>(after_residency.frame_capacity),
        static_cast<unsigned long long>(observation.native_generation),
        static_cast<unsigned long long>(observation.native_nonce),
        static_cast<unsigned long long>(observation.native_submit_count),
        static_cast<unsigned long long>(observation.epoch_submit_count),
        static_cast<unsigned long long>(observation.dispatch_count),
        static_cast<unsigned long long>(observation.tile_dispatch_count),
        static_cast<unsigned long long>(observation.final_count),
        static_cast<unsigned long long>(observation.generated_pages),
        static_cast<unsigned long long>(observation.completed_pages),
        static_cast<unsigned long long>(observation.forecasted_pages),
        static_cast<unsigned long long>(observation.promoted_pages),
        static_cast<unsigned long long>(observation.persisted_pages),
        static_cast<unsigned long long>(observation.wavefront_steps),
        static_cast<unsigned long long>(observation.gpu_read_bytes),
        static_cast<unsigned long long>(observation.gpu_write_bytes),
        static_cast<unsigned long long>(observation.native_completed_ns),
        static_cast<unsigned long long>(observation.native_kernel_ns),
        static_cast<unsigned long long>(observation.native_kernel_samples),
        static_cast<unsigned long long>(observation.native_submit_wait_ns),
        static_cast<unsigned>(observation.host_service),
        static_cast<unsigned>(observation.quarantined),
        static_cast<unsigned long long>(after.command_submits),
        static_cast<unsigned long long>(after.dispatches),
        static_cast<unsigned long long>(after.final_dispatches),
        static_cast<unsigned long long>(after_residency.epoch_count),
        static_cast<unsigned long long>(after_residency.page_in_count),
        static_cast<unsigned long long>(after_residency.page_in_bytes),
        static_cast<unsigned long long>(after_residency.page_out_count),
        static_cast<unsigned long long>(after_residency.page_out_bytes),
        static_cast<unsigned long long>(after_residency.backing_read_bytes),
        static_cast<unsigned long long>(after_residency.backing_write_bytes),
        static_cast<unsigned long long>(after.uploaded_bytes),
        static_cast<unsigned long long>(after.downloaded_bytes),
        static_cast<unsigned long long>(after.external_roundtrip_bytes),
        static_cast<unsigned>(observation.output_match),
        static_cast<unsigned long long>(observation.output_hash),
        static_cast<unsigned long long>(expected_hash));
  }
}

} // namespace

bool validate_dynamic_case(const Case &test_case,
                           const rund::compute::Backend backend,
                           const Observation &observation) noexcept {
  return dynamic_ok(test_case, backend, observation);
}

bool validate_case(const Case &test_case, const rund::compute::Backend backend,
                   const std::span<const Observation> observations) noexcept {
  if (backend != rund::compute::Backend::Metal &&
      backend != rund::compute::Backend::Vulkan) {
    return false;
  }
  const std::uint64_t expected_hash =
      Workload::expected_hash(test_case.variant);
  const oracle_detail::Continuity sequence =
      oracle_detail::continuity(observations);
  bool valid = sequence.all();
  for (std::size_t index = 0u; index < observations.size(); ++index) {
    const Observation &observation = observations[index];
    valid = run_ok(observation, index, expected_hash) && valid;
  }
  const std::shared_ptr<oracle_detail::Owner> retained =
      oracle_detail::owner_of(test_case);
  const bool staged_output =
      !observations.empty() && observations.front().staged_output;
  const bool owner_mode =
      staged_output
          ? retained != nullptr && retained->evidence != nullptr &&
                retained->evidence->public_resident_input_count == 0u &&
                retained->evidence->whole_run_staged_input_count ==
                    InputCount &&
                !retained->evidence->public_resident_output &&
                retained->evidence->whole_run_staged_output
          : retained != nullptr && retained->evidence != nullptr &&
                retained->evidence->public_resident_input_count == InputCount &&
                retained->evidence->whole_run_staged_input_count == 0u &&
                retained->evidence->public_resident_output &&
                !retained->evidence->whole_run_staged_output;
  const bool owner_valid =
      retained != nullptr && retained->proof != nullptr &&
      retained->evidence != nullptr &&
      retained->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphResident &&
      retained->proof->graph_resident.type.scalar ==
          rund::kernel::ComputeScalar::Lane64 &&
      retained->proof->graph_resident.type.domain ==
          rund::kernel::ComputeDomain::U64 &&
      retained->proof->graph_resident.type.element_bytes ==
          sizeof(std::uint64_t) &&
      retained->proof->graph_wavefront.frame_capacity == 2u &&
      retained->proof->graph_wavefront.batch_count == BatchCount &&
      retained->cold_prepare_count == 1u &&
      retained->warm_rearm_count == RunCount - 1u &&
      retained->evidence->public_handoff_count == 1u &&
      retained->evidence->authority_accept_count == 1u &&
      retained->evidence->pipeline_terminal_count == StageCount &&
      retained->evidence->backing_publication_count == 1u &&
      retained->evidence->final_received && !retained->evidence->quarantined &&
      owner_mode;
  if (!valid || !owner_valid) {
    print_diagnostics(observations, expected_hash, valid, retained, owner_valid);
  } else if (!observations.empty()) {
    const Observation &last = observations.back();
    std::fprintf(
        stdout,
        "GraphResident positive native={submit=%llu logical=%llu physical=%llu "
        "steps=%llu trace=%llu host=0 final=%llu publication=1} output=1\n",
        static_cast<unsigned long long>(last.native_submit_count),
        static_cast<unsigned long long>(last.dispatch_count),
        static_cast<unsigned long long>(last.tile_dispatch_count),
        static_cast<unsigned long long>(last.wavefront_steps),
        static_cast<unsigned long long>(
            retained->evidence->native.graph_wavefront_trace),
        static_cast<unsigned long long>(last.final_count));
  }
  return valid && owner_valid;
}

bool validate_u32_case(
    const U32Case &test_case, const rund::compute::Backend backend,
    const std::span<const Observation> observations) noexcept {
  if (backend != rund::compute::Backend::Metal &&
      backend != rund::compute::Backend::Vulkan) {
    return false;
  }
  const std::uint64_t expected_hash = U32Workload::expected_hash();
  const oracle_detail::Continuity sequence =
      oracle_detail::continuity(observations);
  bool valid = sequence.all();
  for (std::size_t index = 0u; index < observations.size(); ++index) {
    valid = u32_run_ok(observations[index], index, expected_hash) && valid;
  }
  const std::shared_ptr<oracle_detail::Owner> retained =
      oracle_detail::owner_of(test_case);
  const bool owner_valid =
      retained != nullptr && retained->proof != nullptr &&
      retained->evidence != nullptr &&
      retained->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphResident &&
      retained->proof->graph_resident.type.scalar ==
          rund::kernel::ComputeScalar::Lane32 &&
      retained->proof->graph_resident.type.domain ==
          rund::kernel::ComputeDomain::U32 &&
      retained->proof->graph_resident.type.element_bytes ==
          sizeof(std::uint32_t) &&
      retained->proof->graph_resident.stage_count == StageCount &&
      retained->proof->graph_resident.resource_count == 8u &&
      retained->proof->graph_wavefront.frame_capacity == 2u &&
      retained->proof->graph_wavefront.batch_count == BatchCount &&
      retained->cold_prepare_count == 1u &&
      retained->warm_rearm_count == RunCount - 1u &&
      retained->evidence->public_handoff_count == 1u &&
      retained->evidence->authority_accept_count == 1u &&
      retained->evidence->pipeline_terminal_count == StageCount &&
      retained->evidence->backing_publication_count == 1u &&
      retained->evidence->public_resident_input_count == InputCount &&
      retained->evidence->whole_run_staged_input_count == 0u &&
      retained->evidence->public_resident_output &&
      !retained->evidence->whole_run_staged_output &&
      retained->evidence->final_received && !retained->evidence->quarantined;
  if (!valid || !owner_valid) {
    const Observation *last =
        observations.empty() ? nullptr : &observations.back();
    std::fprintf(stderr,
                 "GraphResident U32 valid=%u owner=%u type=%u/%u/%u "
                 "submit=%llu dispatch=%llu final=%llu host=%u output=%u\n",
                 static_cast<unsigned>(valid),
                 static_cast<unsigned>(owner_valid),
                 last == nullptr ? 0u : static_cast<unsigned>(last->proof_scalar),
                 last == nullptr ? 0u : static_cast<unsigned>(last->proof_domain),
                 last == nullptr ? 0u : last->proof_element_bytes,
                 static_cast<unsigned long long>(
                     last == nullptr ? 0u : last->native_submit_count),
                 static_cast<unsigned long long>(
                     last == nullptr ? 0u : last->dispatch_count),
                 static_cast<unsigned long long>(
                     last == nullptr ? 0u : last->final_count),
                 static_cast<unsigned>(last != nullptr && last->host_service),
                 static_cast<unsigned>(last != nullptr && last->output_match));
  } else {
    std::fprintf(stdout,
                 "GraphResident U32 positive native={submit=1 dispatch=1 "
                 "final=1 host=0 publication=1} tail=%zu output=1\n",
                 TailElements);
  }
  return valid && owner_valid;
}

} // namespace rund_node_test_virtual::product::graph_resident
