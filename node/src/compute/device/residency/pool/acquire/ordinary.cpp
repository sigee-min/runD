#include "common.hpp"

#include "../../../../size.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::residency::acquire_detail {

std::shared_ptr<Pool>
acquire_ordinary(Registry &registry,
                 const std::shared_ptr<DeviceState> &device,
                 const PoolLayout &layout,
                 RegistryKeepalive &keepalive) noexcept {
  try {
    Transaction transaction{registry, device, layout, keepalive};
    // Reserve both weak registries before any native allocation.  The input
    // arena may be borrowed, but reserving one slot keeps publication
    // allocation-free in either case.
    if (!transaction.reserve_registry(1u) ||
        !transaction.reserve_pending(1u)) {
      return nullptr;
    }
    const FrameTier execution_tier =
        device->backend == Backend::Cpu ? FrameTier::Host : FrameTier::Device;
    const PhysicalClass input_class{.type = layout.input_type,
                                    .format = layout.input_format,
                                    .page_bytes = layout.input_page_bytes,
                                    .tier = execution_tier,
                                    .role = FrameRole::Input};
    std::size_t input_index = 0u;
    if (!transaction.prepare_arena(
            input_class, layout.frame_capacity, true,
            ArenaReuse::OrdinaryInput, ArenaRegistration::OrdinaryBanks,
            input_index)) {
      return nullptr;
    }
    PendingPhysicalArena &input_candidate = transaction.pending(input_index);
    const std::shared_ptr<PhysicalArena> &input_arena = input_candidate.arena;
    const bool new_input_arena = input_candidate.fresh;

    const std::size_t input_width = type_bytes(layout.input_type);
    const std::size_t intermediate_width =
        type_bytes(layout.intermediate_type);
    const std::size_t control_width = type_bytes(layout.control_type);
    const std::size_t output_width = type_bytes(layout.output_type);
    std::uint64_t input_arena_bytes = 0u;
    std::uint64_t intermediate_arena_bytes = 0u;
    std::uint64_t control_arena_bytes = 0u;
    std::uint64_t output_arena_bytes = 0u;
    if (input_width == 0u || output_width == 0u ||
        !kernel::checked::mul(layout.input_page_bytes,
                              input_arena->frame_capacity,
                              input_arena_bytes) ||
        !kernel::checked::mul(layout.output_page_bytes, layout.frame_capacity,
                              output_arena_bytes) ||
        !kernel::checked::mul(layout.intermediate_page_bytes,
                              layout.frame_capacity,
                              intermediate_arena_bytes) ||
        !kernel::checked::mul(layout.control_page_bytes, layout.frame_capacity,
                              control_arena_bytes) ||
        input_arena_bytes / input_width >
            std::numeric_limits<std::size_t>::max() ||
        (intermediate_arena_bytes != 0u &&
         (intermediate_width == 0u ||
          intermediate_arena_bytes / intermediate_width >
              std::numeric_limits<std::size_t>::max())) ||
        (control_arena_bytes != 0u &&
         (control_width == 0u ||
          control_arena_bytes / control_width >
              std::numeric_limits<std::size_t>::max())) ||
        output_arena_bytes / output_width >
            std::numeric_limits<std::size_t>::max()) {
      return nullptr;
    }
    PoolFootprint footprint{};
    if (!project_pool_footprint(layout, device->backend, footprint) ||
        footprint.host_storage_bytes > std::numeric_limits<std::size_t>::max()) {
      return nullptr;
    }
    const std::uint64_t host_storage_bytes = footprint.host_storage_bytes;
    const auto input_storage =
        planned_buffer_storage_bytes(*device, input_arena_bytes);
    const auto output_storage =
        planned_buffer_storage_bytes(*device, output_arena_bytes);
    const auto intermediate_storage =
        intermediate_arena_bytes == 0u
            ? Result<std::uint64_t>::success(0u)
            : planned_buffer_storage_bytes(*device, intermediate_arena_bytes);
    const auto control_storage =
        control_arena_bytes == 0u
            ? Result<std::uint64_t>::success(0u)
            : planned_buffer_storage_bytes(*device, control_arena_bytes);
    std::uint64_t committed_bytes = 0u;
    std::uint64_t input_committed_bytes = 0u;
    if (!input_storage || !intermediate_storage || !control_storage ||
        !output_storage ||
        !kernel::checked::add(*output_storage, *intermediate_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *control_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *intermediate_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *control_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, *output_storage,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, host_storage_bytes,
                              committed_bytes) ||
        !kernel::checked::mul(*input_storage, Pool::BankCount,
                              input_committed_bytes) ||
        (new_input_arena &&
         input_candidate.committed_bytes != input_committed_bytes)) {
      return nullptr;
    }
    storage::Reservation admission;
    if (!transaction.reserve_budget(committed_bytes, admission)) {
      return nullptr;
    }

    auto pool = std::make_shared<Pool>();
    pool->layout = layout;
    pool->device = device;
    pool->input_arena = input_arena;
    pool->input = input_arena->buffers;
    if (!transaction.make_bank_buffers(
            pool->output, layout.output_type,
            static_cast<std::size_t>(output_arena_bytes / output_width),
            *output_storage, BufferKind::Residency)) {
      return nullptr;
    }
    if (intermediate_arena_bytes != 0u &&
        !transaction.make_bank_buffers(
            pool->intermediate, layout.intermediate_type,
            static_cast<std::size_t>(intermediate_arena_bytes /
                                     intermediate_width),
            *intermediate_storage, BufferKind::InputBinding)) {
      return nullptr;
    }
    if (control_arena_bytes != 0u &&
        !transaction.make_bank_buffers(
            pool->control, layout.control_type,
            static_cast<std::size_t>(control_arena_bytes / control_width),
            *control_storage, BufferKind::InputBinding)) {
      return nullptr;
    }
    if (!transaction.make_host_storage(*pool, host_storage_bytes) ||
        !pool->configure_execution()) {
      return nullptr;
    }
    if (device->backend != Backend::Cpu) {
      for (Prefetcher &prefetcher : pool->prefetch) {
        if (!prefetcher.configure(layout.input_page_bytes,
                                  layout.frame_capacity)) {
          return nullptr;
        }
      }
    }

    if (!transaction.register_new_arenas()) {
      return nullptr;
    }
    const std::uint32_t execution_frame_count =
        layout.frame_capacity * Pool::BankCount;
    const std::uint32_t host_input_frame_count =
        device->backend == Backend::Cpu
            ? 0u
            : layout.host_frame_capacity * layout.graph_host_input_count *
                  Pool::BankCount;
    const std::uint32_t host_output_frame_count =
        device->backend == Backend::Cpu
            ? 0u
            : layout.host_output_frame_capacity * Pool::BankCount;
    std::uint32_t first_intermediate_frame = 0u;
    std::uint32_t first_output_frame = 0u;
    std::uint32_t first_host_input_frame = 0u;
    std::uint32_t first_host_output_frame = 0u;
    if (!transaction.register_region(
            execution_tier, FrameRole::Intermediate,
            layout.intermediate_page_bytes == 0u ? 0u : execution_frame_count,
            0u, 0u, first_intermediate_frame) ||
        !transaction.register_region(execution_tier, FrameRole::Output,
                                     execution_frame_count, 0u, 0u,
                                     first_output_frame) ||
        !transaction.register_region(FrameTier::Host, FrameRole::Input,
                                     host_input_frame_count, 0u, 0u,
                                     first_host_input_frame) ||
        !transaction.register_region(FrameTier::Host, FrameRole::Output,
                                     host_output_frame_count, 0u, 0u,
                                     first_host_output_frame)) {
      return nullptr;
    }
    if (!transaction.commit_pool(admission, committed_bytes) ||
        !transaction.commit_arenas()) {
      return nullptr;
    }

    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      pool->input_regions[bank] = input_arena->bank_regions[bank];
      pool->input_regions[bank].count = layout.frame_capacity;
    }
    pool->first_intermediate_frame = first_intermediate_frame;
    pool->intermediate_frame_count =
        layout.intermediate_page_bytes == 0u ? 0u : execution_frame_count;
    pool->first_output_frame = first_output_frame;
    pool->output_frame_count = execution_frame_count;
    pool->first_host_input_frame = first_host_input_frame;
    pool->host_input_frame_count = host_input_frame_count;
    pool->first_host_output_frame = first_host_output_frame;
    pool->host_output_frame_count = host_output_frame_count;
    transaction.publish(pool, std::move(admission), host_storage_bytes);
    return pool;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

} // namespace rund::compute::detail::residency::acquire_detail
