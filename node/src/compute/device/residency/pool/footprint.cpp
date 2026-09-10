#include "../../../buffer/state.hpp"
#include "internal.hpp"

#include "../../../buffer/local.hpp"
#include "../../../pipeline/plan/contract.hpp"
#include "../../../pipeline/residency/model.hpp"
#include "../../../size.hpp"
#include "../../../type.hpp"
#include "../../state.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail::residency {

bool project_pool_footprint(const PoolLayout &layout, const Backend backend,
                            PoolFootprint &result) noexcept {
  result = {};
  if (layout.input_page_bytes == 0u || layout.output_page_bytes == 0u ||
      layout.graph_host_input_count == 0u ||
      layout.host_frame_capacity == 0u ||
      layout.host_output_frame_capacity == 0u) {
    return false;
  }
  if (!kernel::checked::mul(layout.input_page_bytes,
                            layout.host_frame_capacity,
                            result.host_input_bytes) ||
      !kernel::checked::mul(result.host_input_bytes,
                            layout.graph_host_input_count,
                            result.host_input_bytes) ||
      !kernel::checked::mul(layout.output_page_bytes,
                            layout.host_output_frame_capacity,
                            result.host_output_bytes)) {
    result = {};
    return false;
  }
  if (backend == Backend::Cpu) {
    return true;
  }
  if (!kernel::checked::add(result.host_input_bytes,
                            result.host_output_bytes,
                            result.host_storage_bytes) ||
      !kernel::checked::mul(result.host_storage_bytes, Pool::BankCount,
                            result.host_storage_bytes)) {
    result = {};
    return false;
  }
  return true;
}
std::uint64_t Pool::retained_host_bytes() const noexcept {
  std::uint64_t bytes = sizeof(Pool);
  if (accelerator_execution != nullptr) {
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, sizeof(AcceleratorExecutionRing));
  }
  if (graph_persist != nullptr) {
    bytes =
        ::rund::detail::counter::SaturatingAdd(bytes, sizeof(GraphPersistRing));
    for (const Persister &persister : graph_persist->slots) {
      bytes = ::rund::detail::counter::SaturatingAdd(
          bytes, persister.retained_host_bytes());
    }
  }
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, ::rund::detail::counter::SaturatingMultiply(
                 static_cast<std::uint64_t>(graph_owners.capacity()),
                 static_cast<std::uint64_t>(sizeof(PoolPhysicalOwner))));

  std::array<const PhysicalArena *, TiledGraphResourceCapacity + 1u> arenas{};
  std::size_t arena_count = 0u;
  const auto add_arena = [&](const std::shared_ptr<PhysicalArena> &arena) {
    if (arena == nullptr ||
        std::find(arenas.begin(), arenas.begin() + arena_count, arena.get()) !=
            arenas.begin() + arena_count) {
      return;
    }
    if (arena_count == arenas.size()) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return;
    }
    arenas[arena_count++] = arena.get();
    bytes =
        ::rund::detail::counter::SaturatingAdd(bytes, sizeof(PhysicalArena));
  };
  add_arena(input_arena);
  for (const PoolPhysicalOwner &owner : graph_owners) {
    add_arena(owner.arena);
  }

  constexpr std::size_t BufferCapacity = TiledGraphResourceCapacity * 4u + 8u;
  std::array<const BufferState *, BufferCapacity> buffers{};
  std::size_t buffer_count = 0u;
  const auto add_buffer = [&](const std::shared_ptr<BufferState> &buffer) {
    if (buffer == nullptr ||
        std::find(buffers.begin(), buffers.begin() + buffer_count,
                  buffer.get()) != buffers.begin() + buffer_count) {
      return;
    }
    if (buffer_count == buffers.size()) {
      bytes = std::numeric_limits<std::uint64_t>::max();
      return;
    }
    buffers[buffer_count++] = buffer.get();
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, sizeof(BufferState));
  };
  for (std::size_t index = 0u; index < arena_count; ++index) {
    for (const std::shared_ptr<BufferState> &buffer : arenas[index]->buffers) {
      add_buffer(buffer);
    }
  }
  for (const PoolPhysicalOwner &owner : graph_owners) {
    for (const std::shared_ptr<BufferState> &buffer : owner.buffers) {
      add_buffer(buffer);
    }
  }
  for (const auto *const range : {&input, &intermediate, &control, &output}) {
    for (const std::shared_ptr<BufferState> &buffer : *range) {
      add_buffer(buffer);
    }
  }
  for (const Prefetcher &prefetcher : prefetch) {
    bytes = ::rund::detail::counter::SaturatingAdd(
        bytes, prefetcher.retained_host_bytes());
  }
  return bytes;
}
bool graph_pool_footprint(
    const Pool &pool, const std::span<const TiledGraphPhysicalClass> classes,
    std::uint64_t &bytes) noexcept {
  bytes = 0u;
  if (!graph_classes_match(pool, classes) || pool.device == nullptr ||
      pool.registry == nullptr || pool.layout.frame_capacity == 0u) {
    return false;
  }
  const FrameTier expected_tier = pool.device->backend == Backend::Cpu
                                      ? FrameTier::Host
                                      : FrameTier::Device;
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const TiledGraphPhysicalClass declared = classes[index];
    const PoolPhysicalOwner &owner = pool.graph_owners[index];
    const PhysicalArena *const arena = owner.arena.get();
    std::uint64_t view_bytes = 0u;
    const std::size_t element_bytes = type_bytes(declared.type);
    if (arena == nullptr || arena->registry != pool.registry ||
        arena->extent == 0u || owner.view == 0u ||
        owner.frame_capacity < pool.layout.frame_capacity ||
        owner.physical_class.tier != expected_tier || element_bytes == 0u ||
        declared.page_bytes % element_bytes != 0u ||
        !kernel::checked::mul(owner.frame_capacity, declared.page_bytes,
                              view_bytes) ||
        view_bytes != arena->bank_bytes ||
        view_bytes > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    const std::size_t exact_bytes = static_cast<std::size_t>(view_bytes);
    const std::size_t exact_count = exact_bytes / element_bytes;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (pool.graph_owners[prior].arena == owner.arena) {
        return false;
      }
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      const FrameRegion arena_region = arena->bank_regions[bank];
      const FrameRegion cache_region = owner.cache_regions[bank];
      const FrameRegion owner_region = owner.bank_regions[bank];
      if (owner.buffers[bank] == nullptr || arena->buffers[bank] == nullptr ||
          !same_physical_buffer(*owner.buffers[bank], *arena->buffers[bank]) ||
          owner.buffers[bank]->device != pool.device ||
          owner.buffers[bank]->type != declared.type ||
          owner.buffers[bank]->count != exact_count ||
          owner.buffers[bank]->bytes != exact_bytes ||
          owner.buffers[bank]->physical_bytes < exact_bytes ||
          arena_region.tier != expected_tier || arena_region.count == 0u ||
          cache_region.tier != expected_tier ||
          cache_region.role != frame_role(declared.role) ||
          cache_region.count != owner.frame_capacity ||
          arena_region.count != arena->frame_capacity ||
          owner_region.tier != cache_region.tier ||
          owner_region.role != cache_region.role ||
          owner_region.first != cache_region.first ||
          owner_region.count != pool.layout.frame_capacity) {
        return false;
      }
    }
    if (!kernel::checked::add(bytes, declared.page_bytes, bytes)) {
      return false;
    }
  }
  for (const PoolPhysicalOwner &owner : pool.graph_owners) {
    if (!pool.registry->authority().owns_regions(owner.bank_regions) ||
        !pool.registry->authority().owns_regions(owner.cache_regions) ||
        !pool.registry->authority().owns_regions(owner.arena->bank_regions)) {
      return false;
    }
  }
  return bytes != 0u;
}

} // namespace rund::compute::detail::residency
