#include "local.hpp"

#include "../../graph_pointwise_shape/evidence.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"
#include "src/hash/fnv.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

namespace {

using rund::node::accel::detail::DeviceVsmTopology;

[[nodiscard]] std::uint64_t
expected_hash(const ResidentCase &test_case) noexcept {
  return ::rund::node::hash_detail::HashBytes(test_case.expected.data(),
                                              test_case.expected.size() *
                                                  sizeof(std::uint64_t));
}

[[nodiscard]] bool route_ok(const ResidentObservation &observation) noexcept {
  return observation.status && observation.graph_resident &&
         observation.route_kind == RouteKind::DeviceVsm &&
         observation.accepted_owner_mask == OwnerDeviceVsm &&
         observation.accepted_owner_count == 1u;
}

[[nodiscard]] bool proof_ok(const ResidentObservation &observation) noexcept {
  constexpr std::array<std::uint8_t, StageCount> stage_ports{3u, 2u, 2u, 2u,
                                                             2u, 2u, 2u};
  return observation.proof_stage_count == StageCount &&
         observation.proof_resource_count == ResourceCount &&
         observation.proof_port_count == PortCount &&
         observation.proof_scalar ==
             static_cast<std::uint8_t>(rund::kernel::ComputeScalar::Lane64) &&
         observation.proof_domain ==
             static_cast<std::uint8_t>(rund::kernel::ComputeDomain::U64) &&
         observation.proof_element_bytes == sizeof(std::uint64_t) &&
         observation.wavefront_stage_count == StageCount &&
         observation.wavefront_frame_capacity == FrameCapacity &&
         observation.wavefront_batch_count == BatchCount &&
         observation.proof_owner_binding_count == OwnerCount &&
         observation.proof_stage_ports == stage_ports &&
         observation.proof_digest != 0u;
}

[[nodiscard]] bool native_ok(const ResidentObservation &observation) noexcept {
  constexpr std::uint64_t logical_bytes =
      (PageCount * FrameElements - TailElements) * sizeof(std::uint64_t);
  return observation.native_submit_count == 1u &&
         observation.epoch_submit_count == 0u &&
         observation.dispatch_count == 1u &&
         observation.tile_dispatch_count == 1u &&
         observation.final_count == 1u &&
         observation.native_generated_epochs == PageCount &&
         observation.native_completed_epochs == PageCount &&
         observation.native_proof_hi != 0u &&
         observation.native_proof_lo != 0u &&
         observation.native_generation != 0u &&
         observation.native_nonce != 0u &&
         observation.native_gpu_backing_read_bytes == 0u &&
         observation.native_gpu_backing_write_bytes == logical_bytes &&
         observation.wavefront_steps == StageCount * BatchCount &&
         observation.native_may_write && !observation.host_service &&
         !observation.quarantined && observation.final_received &&
         observation.public_handoff_count == 1u &&
         observation.authority_accept_count == 1u &&
         observation.pipeline_terminal_count == StageCount &&
         observation.backing_publication_count == 1u;
}

[[nodiscard]] bool stats_ok(const ResidentObservation &observation) noexcept {
  constexpr std::uint64_t logical_bytes =
      (PageCount * FrameElements - TailElements) * sizeof(std::uint64_t);
  constexpr std::uint64_t payload_bytes = FrameElements * sizeof(std::uint64_t);
  const auto &stats = observation.after;
  const auto &residency = stats.pipeline.residency;
  return stats.command_submits == 1u && stats.dispatches == 1u &&
         stats.final_dispatches == 1u && stats.uploaded_bytes == 0u &&
         stats.downloaded_bytes == 0u && stats.external_roundtrip_bytes == 0u &&
         residency.logical_bytes == ResourceCount * logical_bytes &&
         residency.active_count == PageCount * FrameElements - TailElements &&
         residency.page_bytes == PhysicalCount * payload_bytes &&
         residency.page_count == PageCount &&
         residency.frame_capacity == FrameCapacity &&
         residency.epoch_count == BatchCount &&
         residency.window_handoff_count == 1u &&
         residency.window_batch_count == 1u &&
         residency.window_queue_call_count == 1u &&
         residency.page_in_count == InputCount * PageCount &&
         residency.page_in_bytes == 0u &&
         residency.page_out_count == PageCount &&
         residency.page_out_bytes == logical_bytes &&
         residency.backing_read_bytes == 0u &&
         residency.backing_write_bytes == 0u && residency.backing_io_ns == 0u;
}

[[nodiscard]] bool run_ok(const ResidentObservation &observation,
                          const std::size_t index,
                          const std::uint64_t hash) noexcept {
  return index < RunCount && route_ok(observation) && proof_ok(observation) &&
         native_ok(observation) && stats_ok(observation) &&
         !observation.staged_output && observation.output_match &&
         observation.output_hash == hash &&
         observation.version_after == observation.version_before + 1u &&
         observation.cold_prepare_count == 1u &&
         observation.warm_rearm_count == index;
}

[[nodiscard]] bool
continuity(const std::span<const ResidentObservation> observations) noexcept {
  if (observations.size() != RunCount) {
    return false;
  }
  for (std::size_t index = 0u; index < observations.size(); ++index) {
    const ResidentObservation &current = observations[index];
    for (std::size_t stage = 0u; stage < StageCount; ++stage) {
      if (current.generations_after[stage] <=
              current.generations_before[stage] ||
          current.generations_after[stage] !=
              current.generations_before[stage] + 1u) {
        return false;
      }
    }
    if (index == 0u) {
      continue;
    }
    const ResidentObservation &prior = observations[index - 1u];
    if (current.version_before != prior.version_after ||
        current.proof_digest != prior.proof_digest ||
        current.proof_scalar != prior.proof_scalar ||
        current.proof_domain != prior.proof_domain ||
        current.proof_element_bytes != prior.proof_element_bytes ||
        current.wavefront_stage_count != prior.wavefront_stage_count ||
        current.wavefront_frame_capacity != prior.wavefront_frame_capacity ||
        current.wavefront_batch_count != prior.wavefront_batch_count ||
        current.proof_stage_ports != prior.proof_stage_ports ||
        current.proof_owner_binding_count != prior.proof_owner_binding_count ||
        current.native_proof_hi != prior.native_proof_hi ||
        current.native_proof_lo != prior.native_proof_lo ||
        current.native_generation <= prior.native_generation ||
        current.native_nonce == prior.native_nonce) {
      return false;
    }
    for (std::size_t stage = 0u; stage < StageCount; ++stage) {
      if (current.generations_before[stage] != prior.generations_after[stage]) {
        return false;
      }
    }
  }
  return true;
}

void print_resident_failure(
    const ResidentObservation &observation, const std::size_t index,
    const std::uint64_t hash, const bool all_continuity,
    const ResidentObservation *const previous) noexcept {
  const bool route = route_ok(observation);
  const bool proof = proof_ok(observation);
  const bool native = native_ok(observation);
  const bool stats = stats_ok(observation);
  const bool staged = !observation.staged_output;
  const bool output = observation.output_match;
  const bool hash_match = observation.output_hash == hash;
  const bool version =
      observation.version_after == observation.version_before + 1u;
  const bool cold = observation.cold_prepare_count == 1u;
  const bool warm = observation.warm_rearm_count == index;
  const auto &residency = observation.after.pipeline.residency;

  std::fprintf(
      stderr,
      "Graph depth-seven resident[%zu] bool{route=%u proof=%u native=%u "
      "stats=%u staged=%u output=%u hash=%u version=%u cold=%u warm=%u "
      "continuity=%u}\n",
      index, static_cast<unsigned>(route), static_cast<unsigned>(proof),
      static_cast<unsigned>(native), static_cast<unsigned>(stats),
      static_cast<unsigned>(staged), static_cast<unsigned>(output),
      static_cast<unsigned>(hash_match), static_cast<unsigned>(version),
      static_cast<unsigned>(cold), static_cast<unsigned>(warm),
      static_cast<unsigned>(all_continuity));

  std::fprintf(
      stderr,
      "Graph depth-seven resident[%zu] route{kind=%u owner=%u/%u} "
      "proof{digest=%llu type=%u/%u/%u shape=%u/%u/%u/%u "
      "wave=%u/%u/%u ports=[",
      index, static_cast<unsigned>(observation.route_kind),
      observation.accepted_owner_mask, observation.accepted_owner_count,
      static_cast<unsigned long long>(observation.proof_digest),
      observation.proof_scalar, observation.proof_domain,
      observation.proof_element_bytes, observation.proof_stage_count,
      observation.proof_resource_count, observation.proof_port_count,
      observation.proof_owner_binding_count, observation.wavefront_stage_count,
      observation.wavefront_frame_capacity, observation.wavefront_batch_count);
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    std::fprintf(stderr, "%s%u", stage == 0u ? "" : ",",
                 observation.proof_stage_ports[stage]);
  }
  std::fprintf(stderr,
               "]} output{staged=%u match=%u hash=%llu expected=%llu "
               "version=%llu->%llu}\n",
               static_cast<unsigned>(observation.staged_output),
               static_cast<unsigned>(observation.output_match),
               static_cast<unsigned long long>(observation.output_hash),
               static_cast<unsigned long long>(hash),
               static_cast<unsigned long long>(observation.version_before),
               static_cast<unsigned long long>(observation.version_after));

  std::fprintf(
      stderr,
      "Graph depth-seven resident[%zu] native{sub=%llu epoch=%llu "
      "dispatch=%llu tile=%llu final=%llu generated=%llu completed=%llu "
      "steps=%llu gpu=%llu/%llu proof=%llu:%llu generation=%llu nonce=%llu "
      "may_write=%u host=%u quarantine=%u handoff=%llu authority=%llu "
      "terminal=%llu publication=%llu final_received=%u}\n",
      index, static_cast<unsigned long long>(observation.native_submit_count),
      static_cast<unsigned long long>(observation.epoch_submit_count),
      static_cast<unsigned long long>(observation.dispatch_count),
      static_cast<unsigned long long>(observation.tile_dispatch_count),
      static_cast<unsigned long long>(observation.final_count),
      static_cast<unsigned long long>(observation.native_generated_epochs),
      static_cast<unsigned long long>(observation.native_completed_epochs),
      static_cast<unsigned long long>(observation.wavefront_steps),
      static_cast<unsigned long long>(
          observation.native_gpu_backing_read_bytes),
      static_cast<unsigned long long>(
          observation.native_gpu_backing_write_bytes),
      static_cast<unsigned long long>(observation.native_proof_hi),
      static_cast<unsigned long long>(observation.native_proof_lo),
      static_cast<unsigned long long>(observation.native_generation),
      static_cast<unsigned long long>(observation.native_nonce),
      static_cast<unsigned>(observation.native_may_write),
      static_cast<unsigned>(observation.host_service),
      static_cast<unsigned>(observation.quarantined),
      static_cast<unsigned long long>(observation.public_handoff_count),
      static_cast<unsigned long long>(observation.authority_accept_count),
      static_cast<unsigned long long>(observation.pipeline_terminal_count),
      static_cast<unsigned long long>(observation.backing_publication_count),
      static_cast<unsigned>(observation.final_received));

  std::fprintf(
      stderr,
      "Graph depth-seven resident[%zu] stats{cmd=%llu dispatch=%llu "
      "final=%llu logical=%llu active=%llu page_bytes=%llu pages=%llu "
      "frame=%llu epochs=%llu window=%llu/%llu/%llu "
      "page_in=%llu/%llu page_out=%llu/%llu backing=%llu/%llu io_ns=%llu "
      "xfer=%llu/%llu/%llu}\n",
      index, static_cast<unsigned long long>(observation.after.command_submits),
      static_cast<unsigned long long>(observation.after.dispatches),
      static_cast<unsigned long long>(observation.after.final_dispatches),
      static_cast<unsigned long long>(residency.logical_bytes),
      static_cast<unsigned long long>(residency.active_count),
      static_cast<unsigned long long>(residency.page_bytes),
      static_cast<unsigned long long>(residency.page_count),
      static_cast<unsigned long long>(residency.frame_capacity),
      static_cast<unsigned long long>(residency.epoch_count),
      static_cast<unsigned long long>(residency.window_handoff_count),
      static_cast<unsigned long long>(residency.window_batch_count),
      static_cast<unsigned long long>(residency.window_queue_call_count),
      static_cast<unsigned long long>(residency.page_in_count),
      static_cast<unsigned long long>(residency.page_in_bytes),
      static_cast<unsigned long long>(residency.page_out_count),
      static_cast<unsigned long long>(residency.page_out_bytes),
      static_cast<unsigned long long>(residency.backing_read_bytes),
      static_cast<unsigned long long>(residency.backing_write_bytes),
      static_cast<unsigned long long>(residency.backing_io_ns),
      static_cast<unsigned long long>(observation.after.uploaded_bytes),
      static_cast<unsigned long long>(observation.after.downloaded_bytes),
      static_cast<unsigned long long>(
          observation.after.external_roundtrip_bytes));

  std::fprintf(stderr, "Graph depth-seven resident[%zu] generations{before=[",
               index);
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    std::fprintf(
        stderr, "%s%llu", stage == 0u ? "" : ",",
        static_cast<unsigned long long>(observation.generations_before[stage]));
  }
  std::fprintf(stderr, "] after=[");
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    std::fprintf(
        stderr, "%s%llu", stage == 0u ? "" : ",",
        static_cast<unsigned long long>(observation.generations_after[stage]));
  }
  std::fprintf(stderr, "] valid=[");
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const bool valid = observation.generations_after[stage] >
                           observation.generations_before[stage] &&
                       observation.generations_after[stage] ==
                           observation.generations_before[stage] + 1u;
    std::fprintf(stderr, "%s%u", stage == 0u ? "" : ",",
                 static_cast<unsigned>(valid));
  }
  std::fprintf(stderr, "] link=[");
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    const bool linked =
        previous == nullptr || observation.generations_before[stage] ==
                                   previous->generations_after[stage];
    std::fprintf(stderr, "%s%u", stage == 0u ? "" : ",",
                 static_cast<unsigned>(linked));
  }
  std::fprintf(stderr, "]}\n");
}

} // namespace

bool validate_resident_case(
    const ResidentCase &test_case, const rund::compute::Backend backend,
    const std::span<const ResidentObservation> observations) noexcept {
  if ((backend != rund::compute::Backend::Metal &&
       backend != rund::compute::Backend::Vulkan) ||
      test_case.output == nullptr || observations.size() != RunCount) {
    return false;
  }
  const std::uint64_t hash = expected_hash(test_case);
  bool valid = continuity(observations);
  for (std::size_t index = 0u; index < observations.size(); ++index) {
    valid = run_ok(observations[index], index, hash) && valid;
  }
  const std::shared_ptr<Owner> retained = owner_of(test_case);
  const bool endpoints = [&] {
    if (test_case.inputs[0u] == nullptr || test_case.inputs[1u] == nullptr) {
      return false;
    }
    for (const auto &input : test_case.inputs) {
      if (input == nullptr ||
          rund::compute::detail::VirtualBackingAccess::resident(*input) ==
              nullptr) {
        return false;
      }
    }
    return rund::compute::detail::VirtualBackingAccess::resident(
               *test_case.output) != nullptr;
  }();
  const bool owner_valid =
      endpoints && retained != nullptr && retained->proof != nullptr &&
      retained->evidence != nullptr &&
      retained->proof->topology == DeviceVsmTopology::GraphResident &&
      retained->proof->graph_resident.stage_count == StageCount &&
      retained->proof->graph_resident.resource_count == ResourceCount &&
      retained->proof->graph_resident.port_count == PortCount &&
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
    for (std::size_t index = 0u; index < observations.size(); ++index) {
      print_resident_failure(observations[index], index, hash,
                             continuity(observations),
                             index == 0u ? nullptr : &observations[index - 1u]);
    }
    const ResidentObservation *last =
        observations.empty() ? nullptr : &observations.back();
    std::fprintf(
        stderr,
        "Graph depth-seven resident valid=%u owner=%u route=%u "
        "type=%u/%u/%u shape=%u/%u/%u/%u submit=%llu dispatch=%llu "
        "final=%llu host=%u output=%u\n",
        static_cast<unsigned>(valid), static_cast<unsigned>(owner_valid),
        last == nullptr ? 0u : static_cast<unsigned>(last->route_kind),
        last == nullptr ? 0u : static_cast<unsigned>(last->proof_scalar),
        last == nullptr ? 0u : static_cast<unsigned>(last->proof_domain),
        last == nullptr ? 0u : last->proof_element_bytes,
        last == nullptr ? 0u : last->proof_stage_count,
        last == nullptr ? 0u : last->proof_resource_count,
        last == nullptr ? 0u : last->proof_port_count,
        last == nullptr ? 0u : last->proof_owner_binding_count,
        last == nullptr
            ? 0ull
            : static_cast<unsigned long long>(last->native_submit_count),
        last == nullptr ? 0ull
                        : static_cast<unsigned long long>(last->dispatch_count),
        last == nullptr ? 0ull
                        : static_cast<unsigned long long>(last->final_count),
        last == nullptr ? 0u : static_cast<unsigned>(last->host_service),
        last == nullptr ? 0u : static_cast<unsigned>(last->output_match));
  } else {
    std::fprintf(stdout,
                 "Graph depth-seven resident positive topo=GraphResident "
                 "stages=%zu resources=%zu ports=%zu batch=%zu steps=%zu "
                 "submit=1 dispatch=1 final=1 host=0 tail=%zu output=1\n",
                 StageCount, ResourceCount, PortCount, BatchCount,
                 StageCount * BatchCount, TailElements);
  }
  return valid && owner_valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
