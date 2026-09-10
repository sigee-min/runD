#include "../pool.hpp"

#include "../../state.hpp"

#include <algorithm>
#include <exception>

namespace rund::compute::detail::residency {

PhysicalArena::~PhysicalArena() {
  if (registry != nullptr && !registry->release(*this)) {
    std::terminate();
  }
}

Pool::~Pool() {
  for (const Prefetcher &prefetcher : prefetch) {
    if (!prefetcher.quiescent()) {
      std::terminate();
    }
  }
  if (graph_persist != nullptr &&
      !std::all_of(graph_persist->slots.begin(), graph_persist->slots.end(),
                   [](const Persister &slot) { return slot.quiescent(); })) {
    std::terminate();
  }
  if (registry != nullptr) {
    if (!registry->release(*this)) {
      std::terminate();
    }
  }
  if (host_accounted) {
    std::terminate();
  }
}

bool Registry::release(PhysicalArena &arena) noexcept {
  std::lock_guard lock{gate_};
  if (arena.registry != this || arena.frame_capacity == 0u) {
    return false;
  }
  if (!authority_.release_frames(arena.bank_regions)) {
    return false;
  }
  // The Registry gate keeps a concurrent acquire from reusing either the
  // released frame coordinates or refunded budget until the native payload is
  // gone. Destruction order of class members is not an admission protocol.
  for (std::shared_ptr<BufferState> &buffer : arena.buffers) {
    buffer.reset();
  }
  if (arena.memory && !arena.memory.refund()) {
    return false;
  }
  arena.registry = nullptr;
  arena.bank_regions = {};
  arena.frame_capacity = 0u;
  return true;
}

bool Registry::release(Pool &pool) noexcept {
  std::lock_guard lock{gate_};
  if (pool.device == nullptr) {
    return false;
  }
  std::array<FrameRegion, TiledGraphResourceCapacity * Pool::BankCount + 4u>
      regions{};
  std::size_t count = 0u;
  const bool graph_owned = !pool.graph_owners.empty();
  if (graph_owned) {
    for (const PoolPhysicalOwner &owner : pool.graph_owners) {
      if (!owner.owns_regions) {
        continue;
      }
      for (const FrameRegion region : owner.cache_regions) {
        regions[count++] = region;
      }
    }
  }
  if (!graph_owned && pool.intermediate_frame_count != 0u) {
    regions[count++] = FrameRegion{.tier = pool.device->backend == Backend::Cpu
                                               ? FrameTier::Host
                                               : FrameTier::Device,
                                   .role = FrameRole::Intermediate,
                                   .first = pool.first_intermediate_frame,
                                   .count = pool.intermediate_frame_count};
  }
  if (!graph_owned) {
    regions[count++] = FrameRegion{.tier = pool.device->backend == Backend::Cpu
                                               ? FrameTier::Host
                                               : FrameTier::Device,
                                   .role = FrameRole::Output,
                                   .first = pool.first_output_frame,
                                   .count = pool.output_frame_count};
  }
  if (pool.host_input_frame_count != 0u) {
    regions[count++] = FrameRegion{.tier = FrameTier::Host,
                                   .role = FrameRole::Input,
                                   .first = pool.first_host_input_frame,
                                   .count = pool.host_input_frame_count};
  }
  if (pool.host_output_frame_count != 0u) {
    regions[count++] = FrameRegion{.tier = FrameTier::Host,
                                   .role = FrameRole::Output,
                                   .first = pool.first_host_output_frame,
                                   .count = pool.host_output_frame_count};
  }
  if (count != 0u && !authority_.release_frames(
                         std::span<const FrameRegion>{regions.data(), count})) {
    return false;
  }
  if (pool.graph_persist != nullptr) {
    if (!std::all_of(pool.graph_persist->slots.begin(),
                     pool.graph_persist->slots.end(),
                     [](const Persister &slot) { return slot.quiescent(); })) {
      return false;
    }
    pool.graph_persist.reset();
  }
  // Keep the Registry gate through physical destruction and budget refund.
  // Otherwise a concurrent prepare can consume refunded capacity while these
  // native buffers are still alive, violating the single admission bound.
  for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
    pool.input[bank].reset();
    pool.intermediate[bank].reset();
    pool.control[bank].reset();
    pool.output[bank].reset();
  }
  // Drop Pool-owned payload aliases before its reservation refund, but retain
  // PhysicalArena strong references until member destruction after this gate
  // is released. Last-arena destruction re-enters Registry::release().
  for (PoolPhysicalOwner &owner : pool.graph_owners) {
    for (std::shared_ptr<BufferState> &buffer : owner.buffers) {
      buffer.reset();
    }
  }
  pool.host_storage.reset();
  pool.host_storage_bytes = 0u;
  if (pool.memory && !pool.memory.refund()) {
    return false;
  }
  if (pool.host_accounted) {
    if (pool.device == nullptr) {
      return false;
    }
    std::lock_guard memory_lock{pool.device->memory.gate};
    pool.device->memory.host.current =
        pool.device->memory.host.current >= pool.host_bytes
            ? pool.device->memory.host.current - pool.host_bytes
            : 0u;
    pool.host_accounted = false;
    pool.host_bytes = 0u;
  }
  pool.registry = nullptr;
  pool.input_regions = {};
  pool.first_intermediate_frame = 0u;
  pool.intermediate_frame_count = 0u;
  pool.first_output_frame = 0u;
  pool.output_frame_count = 0u;
  pool.first_host_input_frame = 0u;
  pool.host_input_frame_count = 0u;
  pool.first_host_output_frame = 0u;
  pool.host_output_frame_count = 0u;
  return true;
}

} // namespace rund::compute::detail::residency
