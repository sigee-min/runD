#include "../../../../src/compute/buffer/state.hpp"
#include "local.hpp"

#include "../../../../src/compute/backend.hpp"
#include "../../../../src/compute/device/state.hpp"

#include <rund/compute/abi/observe.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund_node_memory_contract {
namespace {

using rund::compute::Backend;
using rund::compute::MemoryStats;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::AccelDeviceState;
using rund::compute::detail::BufferState;
using rund::compute::detail::DeviceOps;
using rund::compute::detail::DeviceState;
using rund::compute::detail::Type;

Status AllocateThenReject(DeviceState &, BufferState &buffer,
                          const std::size_t scalar_bytes,
                          const std::size_t count, bool,
                          rund::node::accel::detail::BackendBufferMemory,
                          std::uint64_t) noexcept {
  buffer.physical_bytes = scalar_bytes * count + 64u;
  return count == 4u ? Status::success() : Status::fail(Reason::BufferCapacity);
}

const DeviceOps kRejectingDeviceOps{
    .allocate = AllocateThenReject,
};

} // namespace

bool CheckRejectedBufferAccounting() {
  auto device = std::make_shared<DeviceState>();
  device->backend = Backend::Metal;
  device->storage = AccelDeviceState{};
  device->ops = &kRejectingDeviceOps;

  auto retained = rund::compute::detail::make_buffer(device, Type::I32, 4u);
  if (!retained) {
    return false;
  }
  const MemoryStats before = rund::compute::detail::device_memory(device);
  if (before.resident.current != 16u || before.device.current != 80u) {
    return false;
  }

  auto rejected = rund::compute::detail::make_buffer(device, Type::I32, 8u);
  const MemoryStats after = rund::compute::detail::device_memory(device);
  if (rejected || rejected.reason() != Reason::BufferCapacity ||
      after != before) {
    return false;
  }

  retained.value().reset();
  const MemoryStats released = rund::compute::detail::device_memory(device);
  return released.resident.current == 0u && released.device.current == 0u &&
         released.resident.peak == before.resident.peak &&
         released.resident.cumulative == before.resident.cumulative &&
         released.device.peak == before.device.peak &&
         released.device.cumulative == before.device.cumulative;
}

} // namespace rund_node_memory_contract
