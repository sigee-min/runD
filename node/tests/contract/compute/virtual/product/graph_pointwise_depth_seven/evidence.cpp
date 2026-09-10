#include "evidence/local.hpp"

#include "../graph_pointwise_shape/evidence.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <span>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

using rund::node::accel::detail::DeviceVsmTopology;

std::shared_ptr<Owner> owner_of(const ResidentCase &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

std::array<std::uint64_t, StageCount>
stage_generations(const Case &test_case) noexcept {
  std::array<std::uint64_t, StageCount> result{};
  graph_pointwise_shape::stage_generations(test_case.state, result);
  return result;
}

bool validate_case(
    Case &test_case, const rund::compute::Backend backend,
    const rund::compute::Status &status, const std::uint64_t initial_version,
    const std::array<std::uint64_t, StageCount> &initial_generations) noexcept {
  return graph_pointwise_shape::validate(
      graph_pointwise_shape::EvidenceView{
          .label = "depth-seven",
          .input_count = InputCount,
          .stage_count = StageCount,
          .frame_elements = FrameElements,
          .page_count = PageCount,
          .tail_elements = TailElements,
          .frame_capacity = test_case.pipeline.plan().residency.frame_capacity,
          .stats = test_case.pipeline.stats(),
          .state = test_case.state,
          .input_backings = std::span{test_case.input_backings},
          .output_backing = test_case.output_backing,
          .expected = std::span{test_case.expected},
      },
      backend, status, initial_version, initial_generations);
}

void resident_stage_generations(
    const ResidentCase &test_case,
    const std::span<std::uint64_t> result) noexcept {
  graph_pointwise_shape::stage_generations(test_case.state, result);
}

bool capture_resident_run(ResidentCase &test_case,
                          ResidentObservation &observation) noexcept {
  const std::shared_ptr<Owner> retained = owner_of(test_case);
  if (retained != nullptr && retained->proof != nullptr) {
    const auto &proof = *retained->proof;
    const auto &graph = proof.graph_resident;
    observation.graph_resident =
        proof.topology == DeviceVsmTopology::GraphResident;
    observation.proof_digest = graph.digest;
    observation.proof_scalar = static_cast<std::uint8_t>(graph.type.scalar);
    observation.proof_domain = static_cast<std::uint8_t>(graph.type.domain);
    observation.proof_element_bytes = graph.type.element_bytes;
    observation.proof_stage_count = graph.stage_count;
    observation.proof_resource_count = graph.resource_count;
    observation.proof_port_count = graph.port_count;
    observation.proof_owner_binding_count = graph.owner_binding_count;
    for (std::size_t index = 0u; index < StageCount; ++index) {
      observation.proof_stage_ports[index] =
          static_cast<std::uint8_t>(graph.stages[index].port_count);
    }
    observation.wavefront_batch_count = proof.graph_wavefront.batch_count;
    observation.wavefront_stage_count = proof.graph_wavefront.stage_count;
    observation.wavefront_frame_capacity = proof.graph_wavefront.frame_capacity;
  }
  if (retained != nullptr && retained->evidence != nullptr) {
    const auto &evidence = *retained->evidence;
    const auto &native = evidence.native;
    observation.native_generated_epochs = native.generated_epochs;
    observation.native_completed_epochs = native.completed_epochs;
    observation.native_proof_hi = native.proof.hi;
    observation.native_proof_lo = native.proof.lo;
    observation.native_generation = native.generation;
    observation.native_nonce = native.nonce;
    observation.native_gpu_backing_read_bytes = native.gpu_backing_read_bytes;
    observation.native_gpu_backing_write_bytes = native.gpu_backing_write_bytes;
    observation.native_submit_count = native.native_submit_count;
    observation.epoch_submit_count = native.epoch_native_submit_count;
    observation.dispatch_count = native.payload_dispatch_count;
    observation.tile_dispatch_count = native.tile_dispatch_count;
    observation.final_count = native.final_callback_count;
    observation.wavefront_steps = native.graph_wavefront_steps;
    observation.native_may_write = native.may_write;
    observation.host_service = native.host_service_turn_count != 0u ||
                               native.host_epoch_callback_count != 0u ||
                               native.epoch_native_submit_count != 0u;
    observation.quarantined = evidence.quarantined;
    observation.public_handoff_count = evidence.public_handoff_count;
    observation.authority_accept_count = evidence.authority_accept_count;
    observation.pipeline_terminal_count = evidence.pipeline_terminal_count;
    observation.backing_publication_count = evidence.backing_publication_count;
    observation.cold_prepare_count = retained->cold_prepare_count;
    observation.warm_rearm_count = retained->warm_rearm_count;
    observation.final_received = evidence.final_received;
  }
  std::vector<std::uint64_t> observed(test_case.expected.size());
  observation.output_match =
      test_case.output != nullptr &&
      static_cast<bool>(test_case.output->read(
          0u, std::as_writable_bytes(std::span{observed}))) &&
      observed == test_case.expected;
  observation.version_after =
      test_case.output == nullptr
          ? 0u
          : rund::compute::detail::VirtualBackingAccess::version(
                *test_case.output);
  observation.staged_output =
      test_case.output == nullptr ||
      rund::compute::detail::VirtualBackingAccess::resident(
          *test_case.output) == nullptr;
  resident_stage_generations(test_case,
                             std::span{observation.generations_after});
  observation.after = test_case.pipeline.stats();
  observation.output_hash = observation.after.output_hash;
  return true;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
