#include "local.hpp"

#include "src/compute/virtual/state.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rund::measure::compute::virtual_graph_pointwise::run_detail {

void record_status(Facts &facts, const ::rund::compute::Status &status) {
  facts.status_observed = true;
  facts.status_code = status.code();
  facts.status_reason = status.reason();
  facts.status_error = std::string(status.error());
  facts.ok = false;
}

double micros(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

Facts facts(Case &test_case, const Backend backend,
            const ::rund::compute::Status status,
            const virtual_residency::ProductRouteEvidence &route,
            const std::uint64_t version_before,
            const std::uint64_t version_after, const bool output_match,
            const std::vector<std::uint64_t> &values,
            const ::rund::compute::Stats &stats) {
  Facts result{};
  record_status(result, status);
  result.stats = stats;
  result.residency = stats.pipeline.residency;
  result.failure_page = result.residency.failed_page;
  result.route = route;
  result.graph_hi = test_case.graph_hi;
  result.graph_lo = test_case.graph_lo;
  result.version_before = version_before;
  result.version_after = version_after;
  result.output_match = output_match;
  result.output_hash = Spec::hash(std::span{values});
  result.ok = static_cast<bool>(status) && output_match;
  const auto state = ::rund::compute::detail::VirtualPipelineAccess::state(
      *test_case.pipeline);
  if (state == nullptr) {
    return result;
  }
  if (state->failure_log.has()) {
    const auto &failure = state->failure_log.first();
    result.failure_present = true;
    result.failure_phase = static_cast<std::uint8_t>(failure.phase);
    result.failure_check = static_cast<std::uint8_t>(failure.check);
    result.failure_stage = failure.stage;
    result.failure_batch = failure.batch;
  }
  const auto owner =
      std::static_pointer_cast<Owner>(state->device_vsm_product_cache);
  result.owner_present = owner != nullptr;
  result.cpu_reference = backend == Backend::Cpu &&
                         stats.backend == Backend::Cpu && owner == nullptr;
  if (result.cpu_reference) {
    result.ok = result.ok && stats.backend == Backend::Cpu;
    return result;
  }
  if (owner == nullptr || owner->proof == nullptr ||
      owner->evidence == nullptr) {
    return result;
  }
  const auto &proof = *owner->proof;
  const auto &graph = proof.graph_pointwise;
  const auto &map = graph.page_map;
  const auto &native = owner->evidence->native;
  result.final_received = owner->evidence->final_received;
  result.may_write = native.may_write;
  result.owner_identity = reinterpret_cast<std::uintptr_t>(owner.get());
  result.control_identity =
      reinterpret_cast<std::uintptr_t>(owner->registration.get());
  result.graph_pointwise =
      proof.topology ==
      ::rund::node::accel::detail::DeviceVsmTopology::GraphPointwise;
  result.page_map_valid =
      result.graph_pointwise &&
      ::rund::node::accel::detail::device_vsm_page_map_active(map) &&
      ::rund::node::accel::detail::device_vsm_page_map_valid(map,
                                                             Spec::PageCount) &&
      map.words[3u] == Spec::MapRows;
  result.proof_valid =
      result.graph_pointwise &&
      ::rund::node::accel::detail::device_vsm_topology_valid(proof) &&
      graph.topology.external_input_count == Spec::InputCount &&
      graph.stage_count == Spec::StageCount && graph.workgroup_width == 256u &&
      graph.wavefront.frame_capacity == Spec::FrameCapacity &&
      graph.wavefront.batch_count == Spec::BatchCount &&
      proof.geometry.page_count == Spec::PageCount &&
      proof.geometry.logical_bytes ==
          Spec::ElementCount * sizeof(std::uint64_t) &&
      proof.parameter_bytes == 0u && proof.plan.param_bytes == 0u &&
      proof.plan.input_buffer_count == Spec::InputCount &&
      proof.plan.output_buffer_count == 1u && proof.plan.dispatch_count == 1u;
  result.proof_valid = result.proof_valid && result.page_map_valid;
  result.proof_digest =
      ::rund::node::accel::detail::device_vsm_page_map_digest(map);
  result.proof_hi = native.proof.hi;
  result.proof_lo = native.proof.lo;
  result.generation = native.generation;
  result.nonce = native.nonce;
  result.input_count = graph.topology.external_input_count;
  result.stage_count = graph.stage_count;
  result.page_count = static_cast<std::uint32_t>(native.page_count);
  result.tail_elements = Spec::TailElements;
  result.frame_capacity = graph.wavefront.frame_capacity;
  result.batch_count = graph.wavefront.batch_count;
  result.map_rows = map.words[3u];
  result.workgroup_width = graph.workgroup_width;
  result.native_submits = native.native_submit_count;
  result.dispatches = native.payload_dispatch_count;
  result.finals = native.final_callback_count;
  result.epoch_submits = native.epoch_native_submit_count;
  result.generated_pages = native.generated_epochs;
  result.forecasted_pages = native.forecasted_pages;
  result.promoted_pages = native.promoted_pages;
  result.completed_pages = native.completed_epochs;
  result.drained_pages = native.drained_pages;
  result.persisted_pages = native.persisted_pages;
  result.host_turns = native.host_service_turn_count;
  result.host_callbacks = native.host_epoch_callback_count;
  result.gpu_read_bytes = native.gpu_backing_read_bytes;
  result.gpu_write_bytes = native.gpu_backing_write_bytes;
  result.uploaded_bytes = stats.uploaded_bytes;
  result.downloaded_bytes = stats.downloaded_bytes;
  result.backing_read_bytes = stats.pipeline.residency.backing_read_bytes;
  result.backing_write_bytes = stats.pipeline.residency.backing_write_bytes;
  result.quarantined = owner->evidence->quarantined;
  result.ok = result.ok && result.graph_pointwise && result.proof_valid &&
              result.final_received && result.may_write && !result.quarantined;
  return result;
}

} // namespace rund::measure::compute::virtual_graph_pointwise::run_detail
