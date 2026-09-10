#include "internal.hpp"

#include "src/compute/device/residency/execution/device_vsm/registration.hpp"

#include <array>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency::device_vsm_test {

namespace residency = rund::compute::detail::residency;

struct ActualProofOwner final {
  rund::kernel::LoweringArtifact artifact{};
  std::array<rund::kernel::ComputeDispatchWindow, 1u> windows{};
};

struct Publication final {
  std::uint64_t count{};
  bool success{};
};

void Publish(void *const raw, const bool success) noexcept {
  auto *const publication = static_cast<Publication *>(raw);
  if (publication == nullptr) {
    return;
  }
  ++publication->count;
  publication->success = success;
}

[[nodiscard]] rund::kernel::ComputeApi
ApiFor(const rund::compute::Backend backend) noexcept {
  return backend == rund::compute::Backend::Metal
             ? rund::kernel::ComputeApi::Metal
             : rund::kernel::ComputeApi::Vulkan;
}

[[nodiscard]] rund::compute::Target
TargetFor(const rund::compute::Backend backend) noexcept {
  return backend == rund::compute::Backend::Metal
             ? rund::compute::Target::metal()
             : rund::compute::Target::vulkan();
}

[[nodiscard]] std::shared_ptr<accel::DeviceVsmProof>
BuildProof(const rund::kernel::ComputeApi api, const std::uint64_t pages,
           const accel::UploadRoute &input, const accel::UploadRoute &output) {
  constexpr std::uint64_t PayloadElements = 16u;
  constexpr std::uint64_t ElementBytes = sizeof(std::uint32_t);
  const std::uint64_t logical_elements = pages * PayloadElements - 3u;
  auto owner = std::make_shared<ActualProofOwner>();
  owner->artifact = CanonicalArtifact(api);
  if (!accel::TransformDeviceVsmSource(owner->artifact, 1u, 1u)) {
    return {};
  }
  owner->windows[0u] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = PayloadElements,
  };
  auto proof = std::make_shared<accel::DeviceVsmProof>();
  proof->identity = accel::DeviceVsmIdentity{
      .hi = api == rund::kernel::ComputeApi::Metal ? 0x4d4554414c56534dull
                                                   : 0x56554c4b414e5653ull,
      .lo = pages,
  };
  proof->semantic_owner = owner;
  proof->artifact = &owner->artifact;
  proof->plan = rund::kernel::ComputePlan{
      .tile_count = PayloadElements,
      .op_hash_hi = owner->artifact.key.op_hash_hi,
      .op_hash_lo = owner->artifact.key.op_hash_lo,
      .api = api,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
      .input_bytes_per_tile = ElementBytes,
      .output_bytes_per_tile = ElementBytes,
      .bytes_per_tile = 2u * ElementBytes,
      .dispatch_window_tiles = PayloadElements,
      .dispatch_count = 1u,
      .fixed_authoritative = true,
      .ok = true,
      .reason = "ok",
  };
  proof->residents = accel::device_vsm_resident_pair(
      input.resident, input.handle, output.resident, output.handle);
  proof->windows = owner->windows.data();
  proof->geometry = accel::DeviceVsmPageGeometry{
      .logical_bytes = logical_elements * ElementBytes,
      .payload_bytes = PayloadElements * ElementBytes,
      .frame_bytes = PayloadElements * ElementBytes,
      .page_count = pages,
      .element_bytes = ElementBytes,
  };
  proof->output_bytes = proof->geometry.logical_bytes;
  proof->window_count = owner->windows.size();
  proof->width = 2u;
  proof->fixed_common_storage = true;
  return proof;
}

[[nodiscard]] wait_detail::Result
CloseAuthority(wait_detail::Owner &owner,
               const accel::DeviceVsmFinal &final) noexcept {
  if (final.terminal == accel::DeviceVsmTerminal::UnknownMayWrite) {
    return wait_detail::dispose_unknown(owner, final.evidence.completed_epochs);
  }
  residency::DirectRecurrenceFinal prepared{};
  Publication publication{};
  if (owner.registry == nullptr || owner.registration == nullptr ||
      !owner.lease ||
      !owner.registry->authority().direct_recurrences().prepare_direct_recurrence_final(
          owner.lease,
          final.check.ok
              ? rund::compute::Status::success()
              : rund::compute::Status::fail(rund::compute::Reason::DeviceLost),
          residency::execution::TerminalKind::Known, final.evidence.may_write,
          final.evidence.completed_epochs, prepared) ||
      !owner.registry->authority().direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared)) ||
      owner.registration->release_pending(owner.lease) !=
          residency::RegistrationResult::Done ||
      !owner.registration->finish_pending(owner.lease, &publication, Publish) ||
      publication.count != 1u || publication.success != final.check.ok ||
      owner.registration->release() != residency::RegistrationResult::Done) {
    return wait_detail::Result::Failed;
  }
  return wait_detail::Result::Closed;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
