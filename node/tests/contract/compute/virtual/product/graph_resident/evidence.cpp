#include "internal.hpp"
#include "oracle_detail.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <cstdio>
#include <span>

namespace rund_node_test_virtual::product::graph_resident {
namespace {

using namespace rund::node::accel::detail;

[[nodiscard]] bool
alias_reuse(const DeviceVsmGraphResidentProof &proof,
            const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  const DeviceVsmGraphResidentResource *x = nullptr;
  const DeviceVsmGraphResidentResource *w = nullptr;
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const auto &resource = proof.resources[index];
    if (resource.first == 0u && resource.last == 2u) {
      x = &resource;
    }
    if (resource.first == 3u && resource.last == 4u) {
      w = &resource;
    }
  }
  return x != nullptr && w != nullptr && x->owner_slot == w->owner_slot &&
         x->physical_id == w->physical_id && x->color == w->color &&
         wavefront.stage_count == StageCount &&
         (wavefront.same_dispatch[3u] & (std::uint8_t{1u} << 2u)) != 0u &&
         (wavefront.same_release[3u] & (std::uint8_t{1u} << 2u)) == 0u;
}

} // namespace

template <class CaseType, class Scalar>
bool capture_run_impl(CaseType &test_case, Observation &observation) noexcept {
  observation.status_reason =
      static_cast<std::uint32_t>(observation.status.reason());
  const std::shared_ptr<oracle_detail::Owner> retained =
      oracle_detail::owner_of(test_case);
  if (retained != nullptr && retained->proof != nullptr) {
    observation.graph_resident =
        retained->proof->topology == DeviceVsmTopology::GraphResident;
    observation.proof_digest = retained->proof->graph_resident.digest;
    observation.proof_scalar = static_cast<std::uint8_t>(
        retained->proof->graph_resident.type.scalar);
    observation.proof_domain = static_cast<std::uint8_t>(
        retained->proof->graph_resident.type.domain);
    observation.proof_element_bytes =
        retained->proof->graph_resident.type.element_bytes;
    observation.resource_count = retained->proof->graph_resident.resource_count;
    observation.internal_owners =
        retained->proof->graph_resident.owner_binding_count;
    observation.alias_reuse = alias_reuse(retained->proof->graph_resident,
                                          retained->proof->graph_wavefront);
  }
  if (retained != nullptr && retained->evidence != nullptr) {
    const auto &native = retained->evidence->native;
    observation.native_proof_hi = native.proof.hi;
    observation.native_proof_lo = native.proof.lo;
    observation.native_generation = native.generation;
    observation.native_nonce = native.nonce;
    observation.native_submit_count = native.native_submit_count;
    observation.epoch_submit_count = native.epoch_native_submit_count;
    observation.dispatch_count = native.payload_dispatch_count;
    observation.tile_dispatch_count = native.tile_dispatch_count;
    observation.final_count = native.final_callback_count;
    observation.generated_pages = native.generated_epochs;
    observation.completed_pages = native.completed_epochs;
    observation.native_completed_ns = native.completed_ns;
    observation.native_kernel_ns = native.kernel_ns;
    observation.native_kernel_samples = native.kernel_samples;
    observation.native_submit_wait_ns = native.submit_wait_ns;
    observation.forecasted_pages = native.forecasted_pages;
    observation.promoted_pages = native.promoted_pages;
    observation.drained_pages = native.drained_pages;
    observation.persisted_pages = native.persisted_pages;
    observation.wavefront_steps = native.graph_wavefront_steps;
    observation.gpu_read_bytes = native.gpu_backing_read_bytes;
    observation.gpu_write_bytes = native.gpu_backing_write_bytes;
    observation.native_may_write = native.may_write;
    observation.host_service = native.host_service_turn_count != 0u ||
                               native.host_epoch_callback_count != 0u ||
                               native.epoch_native_submit_count != 0u;
    observation.quarantined = retained->evidence->quarantined;
  }
  std::vector<Scalar> observed(test_case.expected.size());
  observation.output_match =
      static_cast<bool>(test_case.output) &&
      static_cast<bool>(test_case.output->read(
          0u, std::as_writable_bytes(std::span{observed}))) &&
      observed == test_case.expected;
  observation.version_after =
      rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
  observation.after = test_case.pipeline.stats();
  observation.output_hash = observation.after.output_hash;
  if (!observation.status || !observation.graph_resident ||
      !observation.output_match) {
    std::fprintf(
        stderr,
        "GraphResident run status=%u prepare=%s native={proof=%llu:%llu "
        "gen=%llu "
        "nonce=%llu submit=%llu dispatch=%llu tile=%llu final=%llu "
        "time=%llu/%llu/%llu/%llu may_write=%u} "
        "output=%u\n",
        observation.status_reason,
        observation.device_vsm_prepare_reason == nullptr
            ? "none"
            : observation.device_vsm_prepare_reason,
        static_cast<unsigned long long>(observation.native_proof_hi),
        static_cast<unsigned long long>(observation.native_proof_lo),
        static_cast<unsigned long long>(observation.native_generation),
        static_cast<unsigned long long>(observation.native_nonce),
        static_cast<unsigned long long>(observation.native_submit_count),
        static_cast<unsigned long long>(observation.dispatch_count),
        static_cast<unsigned long long>(observation.tile_dispatch_count),
        static_cast<unsigned long long>(observation.final_count),
        static_cast<unsigned long long>(observation.native_completed_ns),
        static_cast<unsigned long long>(observation.native_kernel_ns),
        static_cast<unsigned long long>(observation.native_kernel_samples),
        static_cast<unsigned long long>(observation.native_submit_wait_ns),
        static_cast<unsigned>(observation.native_may_write),
        static_cast<unsigned>(observation.output_match));
  }
  return true;
}

bool capture_run(Case &test_case, Observation &observation) noexcept {
  return capture_run_impl<Case, std::uint64_t>(test_case, observation);
}

bool capture_u32_run(U32Case &test_case, Observation &observation) noexcept {
  return capture_run_impl<U32Case, std::uint32_t>(test_case, observation);
}

} // namespace rund_node_test_virtual::product::graph_resident
