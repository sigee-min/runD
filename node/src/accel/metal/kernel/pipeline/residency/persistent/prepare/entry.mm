#include "../../../../../../kernel/prepared/interface/api.hpp"
#include "../../../../../../kernel/prepared/model.hpp"
#include "../internal.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

MetalPersistentResidencySlidingPreparation
PrepareMetalPersistentResidencySliding(
    const PersistentResidencySlidingRequest &request) noexcept {
  MetalPersistentResidencySlidingPreparation preparation{};
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MetalAdapter *adapter = nullptr;
    const rund::AccelCheck check =
        metal_persistent_sliding::preflight(request, adapter);
    if (!check.ok) {
      preparation.capability.check = check;
      return preparation;
    }
    std::shared_ptr<metal_persistent_sliding::Owner> owner;
    try {
      owner = std::make_shared<metal_persistent_sliding::Owner>();
    } catch (const std::bad_alloc &) {
      preparation.capability.check = {false, "compute_pipeline_capacity"};
      return preparation;
    } catch (const std::length_error &) {
      preparation.capability.check = {false, "compute_pipeline_capacity"};
      return preparation;
    }
    owner->adapter = adapter;
    owner->owner_nonce = request.owner_nonce;
    owner->prepared = request;
    owner->prepared.lowering.reset();
    id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter->device.get();
    id<MTLCommandQueue> const queue =
        (__bridge id<MTLCommandQueue>)adapter->queue.get();
    owner->command = queue == nil ? nil : [queue commandBuffer];
    owner->done = device == nil ? nil : [device newSharedEvent];
    for (std::size_t slot = 0u; slot < request.width; ++slot) {
      owner->ready[slot] = device == nil ? nil : [device newSharedEvent];
    }
    if (owner->command == nil || owner->done == nil) {
      preparation.capability.check = {false, "accel_metal_command_unavailable"};
      return preparation;
    }
    for (std::size_t slot = 0u; slot < request.width; ++slot) {
      if (owner->ready[slot] == nil) {
        preparation.capability.check = {false,
                                        "accel_metal_command_unavailable"};
        return preparation;
      }
    }
    const rund::AccelCheck encoded = metal_persistent_sliding::encode(*owner);
    if (!encoded.ok) {
      preparation.capability.check = encoded;
      return preparation;
    }
    if (metal_persistent_sliding::SlotCount >
        std::numeric_limits<std::uint64_t>::max() /
            sizeof(metal_persistent_sliding::Coordinate)) {
      preparation.capability.check = {false, "compute_pipeline_capacity"};
      return preparation;
    }
    const std::uint64_t coordinate_bytes =
        static_cast<std::uint64_t>(metal_persistent_sliding::SlotCount) *
        sizeof(metal_persistent_sliding::Coordinate);
    if (coordinate_bytes > std::numeric_limits<std::uint64_t>::max() -
                               sizeof(metal_persistent_sliding::Owner)) {
      preparation.capability.check = {false, "compute_pipeline_capacity"};
      return preparation;
    }
    owner->encoded_intermediate_bytes = coordinate_bytes;
    owner->capability = PersistentResidencySlidingCapability{
        .check = {true, "ok"},
        .memory = request.memory,
        .retained_bytes =
            sizeof(metal_persistent_sliding::Owner) + coordinate_bytes,
        .transient_bytes = sizeof(MetalResidencySlidingPayload),
        .width = request.width,
        .whole_run_preencoded = true,
        .device_generated_recurrence = false,
        .fixed_native_storage = false,
        .fixed_common_storage = false,
        .host_epoch_callbacks_zero = true,
        .mode = request.mode,
    };
    preparation.capability = owner->capability;
    preparation.service = MetalPersistentResidencySlidingServiceOps();
    preparation.encoded_coordinate_count = request.coordinate_count;
    preparation.encoded_intermediate_bytes = coordinate_bytes;
    preparation.lowering = std::move(owner);
  }
#else
  static_cast<void>(request);
#endif
  return preparation;
}

#endif

} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<Owner> owner_of(const std::shared_ptr<void> &opaque) noexcept {
  auto owner = std::static_pointer_cast<Owner>(opaque);
  return owner != nullptr && owner->magic == OwnerMagic ? std::move(owner)
                                                        : nullptr;
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
