#include "local.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency {
namespace {

namespace accel = rund::node::accel::detail;
namespace test = device_vsm_test;

[[nodiscard]] bool CapabilityBoundary() noexcept {
  auto capability = test::Capability();
  if (!accel::device_vsm_capable(capability)) {
    return false;
  }
  capability.device_generated_recurrence = false;
  if (accel::device_vsm_capable(capability)) {
    return false;
  }
  capability.device_generated_recurrence = true;
  capability.physical_ring_storage = false;
  if (accel::device_vsm_capable(capability)) {
    return false;
  }
  capability.physical_ring_storage = true;
  capability.host_service_turns_zero = false;
  return !accel::device_vsm_capable(capability);
}

[[nodiscard]] bool Aggregate(const std::uint64_t pages) {
  auto state = std::make_shared<test::FakeState>();
  state->capability = test::Capability();
  test::Wait wait{};
  test::BuildRequest(wait, state, pages);
  accel::DeviceVsmPreparation preparation{
      .capability = state->capability,
      .lowering = state,
      .submit = test::FakeSubmit,
      .rearm = test::FakeRearm,
  };
  if (!preparation ||
      !accel::device_vsm_request_valid(preparation.capability, wait.request) ||
      !preparation.submit(wait.request).ok || !wait.valid ||
      wait.final_count != 1u || state->submit_count != 1u ||
      state->final_count != 1u || wait.final.evidence.page_count != pages ||
      wait.final.evidence.generated_epochs != pages ||
      wait.final.evidence.completed_epochs != pages ||
      wait.final.evidence.native_submit_count != 1u ||
      wait.final.evidence.epoch_native_submit_count != 0u ||
      wait.final.evidence.payload_dispatch_count != 1u ||
      wait.final.evidence.host_service_turn_count != 0u ||
      wait.final.evidence.host_epoch_callback_count != 0u ||
      wait.final.evidence.final_callback_count != 1u ||
      wait.final.evidence.max_live_frames != 2u) {
    return false;
  }
  if (preparation.submit(wait.request).ok) {
    return false;
  }
  auto malformed = wait.final;
  ++malformed.evidence.host_service_turn_count;
  if (accel::device_vsm_final_valid(wait.request, malformed)) {
    return false;
  }
  malformed = wait.final;
  ++malformed.evidence.canonical_footprint_checksum;
  return !accel::device_vsm_final_valid(wait.request, malformed);
}

[[nodiscard]] bool RearmBoundary() {
  auto state = std::make_shared<test::FakeState>();
  state->capability = test::Capability();
  test::Wait wait{};
  test::BuildRequest(wait, state, 5u);
  accel::DeviceVsmPreparation preparation{
      .capability = state->capability,
      .lowering = state,
      .submit = test::FakeSubmit,
      .rearm = test::FakeRearm,
  };
  if (!preparation) {
    return false;
  }
  state->rearm_check = {false, "compute_backend_unsupported"};
  state->rearm_mutated = false;
  const accel::DeviceVsmRearmResult declined =
      preparation.rearm(preparation.lowering, wait.request.proof);
  if (declined.check.ok || declined.mutated) {
    return false;
  }
  state->rearm_check = {false, "compute_device_lost"};
  state->rearm_mutated = true;
  const accel::DeviceVsmRearmResult lost =
      preparation.rearm(preparation.lowering, wait.request.proof);
  if (lost.check.ok || !lost.mutated) {
    return false;
  }
  state->rearm_check = {true, "ok"};
  const accel::DeviceVsmRearmResult reset =
      preparation.rearm(preparation.lowering, wait.request.proof);
  return reset.check.ok && reset.mutated;
}

[[nodiscard]] bool RejectsDuplicateResource() {
  auto proof = test::Proof(5u);
  proof->residents.rows[1u].backing.id = proof->residents.rows[0u].backing.id;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsDuplicateOwner() {
  auto proof = test::Proof(5u);
  proof->residents.rows[1u].handle = proof->residents.rows[0u].handle;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsMissingOwner() {
  auto proof = test::Proof(5u);
  proof->residents.rows[0u].handle.reset();
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsNoncanonicalRoleOrder() {
  auto proof = test::Proof(5u);
  proof->residents.rows[0u].role = accel::DeviceVsmResidentRole::Output;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsCountMismatch() {
  auto proof = test::Proof(5u);
  proof->residents.count = 1u;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsCapacityOverflow() {
  auto proof = test::Proof(5u);
  proof->residents.count =
      static_cast<std::uint32_t>(accel::DeviceVsmResidentCapacity + 1u);
  proof->residents.input_count =
      static_cast<std::uint32_t>(accel::DeviceVsmResidentCapacity);
  proof->residents.output_count = 1u;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool RejectsNonemptyUnusedRow() {
  auto proof = test::Proof(5u);
  proof->residents.rows[2u].backing = proof->residents.rows[0u].backing;
  return !accel::device_vsm_proof_valid(*proof);
}

[[nodiscard]] bool ProofBoundary() {
  const auto proof = test::Proof(5u);
  return accel::device_vsm_proof_valid(*proof) && RejectsDuplicateResource() &&
         RejectsDuplicateOwner() && RejectsMissingOwner() &&
         RejectsNoncanonicalRoleOrder() && RejectsCountMismatch() &&
         RejectsCapacityOverflow() && RejectsNonemptyUnusedRow();
}

} // namespace

int CheckDeviceVsm() {
  if (!test::CheckSurface()) {
    return 1;
  }
  if (!test::CheckGeometry()) {
    return 2;
  }
  if (!test::CheckProductEvidence()) {
    return 3;
  }
  if (!CapabilityBoundary()) {
    return 4;
  }
  if (!ProofBoundary()) {
    return 5;
  }
  if (!test::CheckAuthority()) {
    return 6;
  }
  if (!test::CheckSource()) {
    return 7;
  }
  if (!test::CheckGraphPointwiseSource()) {
    return 8;
  }
  if (!test::CheckScanSource()) {
    return 9;
  }
  if (!test::CheckGraphMapScanSource()) {
    return 10;
  }
  if (!test::CheckGraphTerminal()) {
    return 11;
  }
  if (!test::CheckReduceSource()) {
    return 12;
  }
  if (!test::CheckReduceTerminal()) {
    return 13;
  }
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    if (!Aggregate(pages)) {
      std::fprintf(stderr, "device VSM aggregate q=%llu failed\n",
                   static_cast<unsigned long long>(pages));
      return 14;
    }
  }
  return RearmBoundary() ? 0 : 15;
}

} // namespace rund_node_test_pipeline_residency
