#include "../backend.hpp"
#include "local.hpp"

namespace rund::compute::detail {

MemoryStats device_memory(const std::shared_ptr<DeviceState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  MemoryStats stats{
      .backend = state->backend,
      .scope = MemoryScope::Backend,
      .host = meter_memory(state->memory.host),
      .device = meter_memory(state->memory.device),
      .transfer = meter_memory(state->memory.transfer),
  };
  const AccelDeviceState *const accel = accel_device(*state);
  if (accel != nullptr) {
    stats.device.budget = accel->pick.caps.device_bytes;
    stats.transfer.budget = accel->pick.caps.device_bytes;
    if (state->ops != nullptr && state->ops->device_staging != nullptr) {
      stats.staging = state->ops->device_staging(*state);
    }
  }
  return stats;
}

MemorySnapshot
device_memory_snapshot(const std::shared_ptr<DeviceState> &state,
                       const std::span<MemoryEntry> entries) noexcept {
  const MemoryStats stats = device_memory(state);
  SnapshotWriter writer{stats, entries};
  if (state == nullptr) {
    return writer.finish();
  }
  writer.add(MemoryCategory::Host, MemoryUse::Metadata, 0u, stats.host);
  writer.add(MemoryCategory::Frame, MemoryUse::Coordinator, 0u, stats.frame);
  writer.add(MemoryCategory::Tile, MemoryUse::Scratch, 0u, stats.tile);
  writer.add(MemoryCategory::Resident, MemoryUse::Internal, 0u, stats.resident);
  writer.add(MemoryCategory::Staging, MemoryUse::Scratch, 0u, stats.staging);
  writer.add(MemoryCategory::Device, MemoryUse::Internal, 0u, stats.device);
  writer.add(MemoryCategory::Transfer, MemoryUse::Traffic, 0u, stats.transfer);
  return writer.finish();
}

} // namespace rund::compute::detail
