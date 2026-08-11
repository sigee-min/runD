#include "residency_pool.hpp"

#include "../buffer/local.hpp"
#include "../size.hpp"
#include "../type.hpp"
#include "state.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

Pool::~Pool() {
  if (!host_accounted || device == nullptr || host_bytes == 0u) {
    return;
  }
  std::lock_guard lock{device->memory.gate};
  device->memory.host.current = device->memory.host.current >= host_bytes
                                    ? device->memory.host.current - host_bytes
                                    : 0u;
}

std::shared_ptr<Pool>
Registry::acquire(const std::shared_ptr<DeviceState> &device,
                  const PoolLayout &layout) noexcept {
  if (device == nullptr || layout.input_page_bytes == 0u ||
      layout.output_page_bytes == 0u || layout.frame_capacity == 0u) {
    return nullptr;
  }
  const std::size_t input_width = type_bytes(layout.input_type);
  const std::size_t output_width = type_bytes(layout.output_type);
  if (input_width == 0u || output_width == 0u ||
      layout.input_page_bytes % input_width != 0u ||
      layout.output_page_bytes % output_width != 0u) {
    return nullptr;
  }
  std::lock_guard lock{gate_};
  for (auto cursor = pools_.begin(); cursor != pools_.end();) {
    if (auto pool = cursor->lock()) {
      if (pool->layout == layout) {
        return pool;
      }
      ++cursor;
    } else {
      cursor = pools_.erase(cursor);
    }
  }
  try {
    std::uint64_t input_arena_bytes = 0u;
    std::uint64_t output_arena_bytes = 0u;
    std::uint64_t staging_bytes = 0u;
    if (!kernel::checked::mul(layout.input_page_bytes, layout.frame_capacity,
                              input_arena_bytes) ||
        !kernel::checked::mul(layout.output_page_bytes, layout.frame_capacity,
                              output_arena_bytes) ||
        !kernel::checked::add(input_arena_bytes, output_arena_bytes,
                              staging_bytes) ||
        input_arena_bytes / input_width >
            std::numeric_limits<std::size_t>::max() ||
        output_arena_bytes / output_width >
            std::numeric_limits<std::size_t>::max() ||
        staging_bytes > std::numeric_limits<std::size_t>::max()) {
      return nullptr;
    }
    const auto input_storage =
        planned_buffer_storage_bytes(*device, input_arena_bytes);
    const auto output_storage =
        planned_buffer_storage_bytes(*device, output_arena_bytes);
    std::uint64_t committed_bytes = 0u;
    std::uint64_t prefetch_bytes = 0u;
    std::uint64_t host_retained_bytes = 0u;
    if (!input_storage || !output_storage ||
        !kernel::checked::mul(input_arena_bytes, 2u, prefetch_bytes) ||
        !kernel::checked::add(staging_bytes, prefetch_bytes,
                              host_retained_bytes) ||
        !kernel::checked::add(*input_storage, *output_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *input_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *output_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, staging_bytes,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, prefetch_bytes,
                              committed_bytes)) {
      return nullptr;
    }
    storage::Reservation admission =
        device->pipeline_memory_budget.reserve(committed_bytes);
    if (!admission) {
      return nullptr;
    }
    auto cache_input = make_planned_input_binding_buffer(
        device, layout.input_type,
        static_cast<std::size_t>(input_arena_bytes / input_width),
        *input_storage);
    auto cache_output = make_planned_input_binding_buffer(
        device, layout.output_type,
        static_cast<std::size_t>(output_arena_bytes / output_width),
        *output_storage);
    auto input = make_planned_input_binding_buffer(
        device, layout.input_type,
        static_cast<std::size_t>(input_arena_bytes / input_width),
        *input_storage);
    auto output = make_planned_input_binding_buffer(
        device, layout.output_type,
        static_cast<std::size_t>(output_arena_bytes / output_width),
        *output_storage);
    if (!cache_input || !cache_output || !input || !output) {
      return nullptr;
    }
    auto pool = std::make_shared<Pool>();
    pool->layout = layout;
    pool->device = device;
    pool->cache_input = std::move(cache_input).value();
    pool->cache_output = std::move(cache_output).value();
    pool->input = std::move(input).value();
    pool->output = std::move(output).value();
    pool->staging =
        std::make_unique<std::byte[]>(static_cast<std::size_t>(staging_bytes));
    pool->staging_bytes = staging_bytes;
    if (!pool->authority.configure(layout.input_page_bytes,
                                   layout.frame_capacity)) {
      return nullptr;
    }
    std::uint64_t configured_prefetch_bytes = 0u;
    for (Prefetcher &prefetcher : pool->prefetch) {
      if (!prefetcher.configure(layout.input_page_bytes,
                                layout.frame_capacity)) {
        return nullptr;
      }
      configured_prefetch_bytes = ::rund::detail::counter::SaturatingAdd(
          configured_prefetch_bytes, prefetcher.storage_bytes());
    }
    if (configured_prefetch_bytes != prefetch_bytes) {
      return nullptr;
    }
    const storage::Status committed = admission.commit(storage::Usage{
        .physical_bytes = committed_bytes,
        .allocated_bytes = committed_bytes,
    });
    if (!committed) {
      return nullptr;
    }
    pool->memory = std::move(admission);
    pool->host_bytes = host_retained_bytes;
    {
      std::lock_guard memory_lock{device->memory.gate};
      device->memory.host.current = ::rund::detail::counter::SaturatingAdd(
          device->memory.host.current, pool->host_bytes);
      device->memory.host.peak =
          std::max(device->memory.host.peak, device->memory.host.current);
      device->memory.host.cumulative = ::rund::detail::counter::SaturatingAdd(
          device->memory.host.cumulative, pool->host_bytes);
    }
    pool->host_accounted = true;
    pools_.emplace_back(pool);
    return pool;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

} // namespace rund::compute::detail::residency
