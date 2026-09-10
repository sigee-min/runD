#include "../../../../buffer/state.hpp"
#include "common.hpp"

#include "../internal.hpp"

#include "../../../../pipeline/residency/model.hpp"
#include "../../../../size.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::residency::acquire_detail {

std::shared_ptr<Pool>
acquire_graph(Registry &registry, const std::shared_ptr<DeviceState> &device,
              const PoolLayout &layout,
              const std::span<const TiledGraphPhysicalClass> graph_classes,
              RegistryKeepalive &keepalive) noexcept {
  try {
    Transaction transaction{registry, device, layout, keepalive};
    if (!transaction.reserve_registry(graph_classes.size()) ||
        !transaction.reserve_pending(graph_classes.size())) {
      return nullptr;
    }
    const FrameTier execution_tier =
        device->backend == Backend::Cpu ? FrameTier::Host : FrameTier::Device;
    const std::size_t control_width = type_bytes(layout.control_type);
    if (control_width == 0u) {
      return nullptr;
    }

    for (const TiledGraphPhysicalClass &declared : graph_classes) {
      const PhysicalClass physical{
          .type = declared.type,
          .format = declared.format,
          .page_bytes = declared.page_bytes,
          .tier = execution_tier,
          .role = frame_role(declared.role),
      };
      std::size_t ignored_index = 0u;
      if (!transaction.prepare_arena(
              physical, layout.frame_capacity,
              declared.role == GraphResourceRole::Input ||
                  declared.role == GraphResourceRole::Output,
              ArenaReuse::GraphExtent, ArenaRegistration::GraphContiguous,
              ignored_index)) {
        return nullptr;
      }
    }

    std::uint64_t control_arena_bytes = 0u;
    if (!kernel::checked::mul(layout.control_page_bytes,
                              layout.frame_capacity, control_arena_bytes)) {
      return nullptr;
    }
    PoolFootprint footprint{};
    if (!project_pool_footprint(layout, device->backend, footprint) ||
        footprint.host_storage_bytes > std::numeric_limits<std::size_t>::max()) {
      return nullptr;
    }
    const std::uint64_t host_storage_bytes = footprint.host_storage_bytes;
    const auto control_storage =
        planned_buffer_storage_bytes(*device, control_arena_bytes);
    std::uint64_t committed_bytes = 0u;
    if (!control_storage ||
        !kernel::checked::mul(*control_storage, Pool::BankCount,
                              committed_bytes) ||
        !kernel::checked::add(committed_bytes, host_storage_bytes,
                              committed_bytes)) {
      return nullptr;
    }
    storage::Reservation pool_admission;
    if (!transaction.reserve_budget(committed_bytes, pool_admission)) {
      return nullptr;
    }

    auto pool = std::make_shared<Pool>();
    pool->layout = layout;
    pool->device = device;
    pool->graph_owners.resize(graph_classes.size());
    for (std::size_t index = 0u; index < graph_classes.size(); ++index) {
      const TiledGraphPhysicalClass &declared = graph_classes[index];
      const std::shared_ptr<PhysicalArena> &arena =
          transaction.pending(index).arena;
      PoolPhysicalOwner &owner = pool->graph_owners[index];
      owner.physical_id = declared.physical_id;
      owner.color = declared.color;
      owner.physical_class = PhysicalClass{
          .type = declared.type,
          .format = declared.format,
          .page_bytes = declared.page_bytes,
          .tier = execution_tier,
          .role = frame_role(declared.role),
      };
      owner.arena = arena;
      const bool canonical_view =
          arena->physical_class.page_bytes == declared.page_bytes &&
          arena->physical_class.tier == execution_tier &&
          arena->physical_class.role == frame_role(declared.role);
      owner.view = canonical_view ? arena->view : transaction.next_view_id();
      if (owner.view == 0u) {
        return nullptr;
      }
      owner.owns_regions = !canonical_view;
      if (arena->bank_bytes == 0u ||
          arena->bank_bytes % declared.page_bytes != 0u ||
          arena->bank_bytes / declared.page_bytes < layout.frame_capacity ||
          arena->bank_bytes / declared.page_bytes >
              std::numeric_limits<std::uint32_t>::max()) {
        return nullptr;
      }
      owner.frame_capacity = static_cast<std::uint32_t>(
          arena->bank_bytes / declared.page_bytes);
      const std::uint64_t view_bytes = arena->bank_bytes;
      const std::size_t view_width = type_bytes(declared.type);
      if (view_width == 0u || view_bytes % view_width != 0u ||
          view_bytes / view_width > std::numeric_limits<std::size_t>::max()) {
        return nullptr;
      }
      for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
        const std::shared_ptr<BufferState> &root = arena->buffers[bank];
        if (root == nullptr) {
          return nullptr;
        }
        if (root->type == declared.type && root->bytes == view_bytes) {
          owner.buffers[bank] = root;
        } else {
          auto view = make_physical_buffer_view(
              root, declared.type,
              static_cast<std::size_t>(view_bytes / view_width));
          if (!view) {
            return nullptr;
          }
          owner.buffers[bank] = std::move(view).value();
        }
      }
      if (declared.role == GraphResourceRole::Input &&
          pool->input[0] == nullptr) {
        pool->input_arena = arena;
        pool->input = owner.buffers;
      } else if (declared.role == GraphResourceRole::Intermediate &&
                 pool->intermediate[0] == nullptr) {
        pool->intermediate = owner.buffers;
      } else if (declared.role == GraphResourceRole::Output &&
                 pool->output[0] == nullptr) {
        pool->output = owner.buffers;
      }
    }
    if (pool->input[0] == nullptr || pool->intermediate[0] == nullptr ||
        pool->output[0] == nullptr) {
      return nullptr;
    }

    std::array<std::shared_ptr<BufferState>, Pool::BankCount> control{};
    if (!transaction.make_bank_buffers(
            control, layout.control_type,
            static_cast<std::size_t>(control_arena_bytes / control_width),
            *control_storage, BufferKind::InputBinding)) {
      return nullptr;
    }
    pool->control = std::move(control);
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
    for (PoolPhysicalOwner &owner : pool->graph_owners) {
      if (!owner.owns_regions) {
        owner.cache_regions = owner.arena->bank_regions;
        continue;
      }
      std::uint32_t first = 0u;
      if (owner.frame_capacity >
              std::numeric_limits<std::uint32_t>::max() / Pool::BankCount ||
          !transaction.register_region(
              execution_tier, owner.physical_class.role,
              owner.frame_capacity * Pool::BankCount, owner.arena->extent,
              owner.view, first)) {
        return nullptr;
      }
      for (std::uint32_t bank = 0u; bank < Pool::BankCount; ++bank) {
        owner.cache_regions[bank] = FrameRegion{
            .tier = execution_tier,
            .role = owner.physical_class.role,
            .first = first + bank * owner.frame_capacity,
            .count = owner.frame_capacity,
        };
      }
    }

    const std::uint32_t host_input_frame_count =
        device->backend == Backend::Cpu
            ? 0u
            : layout.host_frame_capacity * layout.graph_host_input_count *
                  Pool::BankCount;
    const std::uint32_t host_output_frame_count =
        device->backend == Backend::Cpu
            ? 0u
            : layout.host_output_frame_capacity * Pool::BankCount;
    std::uint32_t first_host_input_frame = 0u;
    std::uint32_t first_host_output_frame = 0u;
    if (!transaction.register_region(FrameTier::Host, FrameRole::Input,
                                     host_input_frame_count, 0u, 0u,
                                     first_host_input_frame) ||
        !transaction.register_region(FrameTier::Host, FrameRole::Output,
                                     host_output_frame_count, 0u, 0u,
                                     first_host_output_frame)) {
      return nullptr;
    }
    if (!transaction.commit_pool(pool_admission, committed_bytes) ||
        !transaction.commit_arenas()) {
      return nullptr;
    }

    for (PoolPhysicalOwner &owner : pool->graph_owners) {
      for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
        if (owner.cache_regions[bank].count == 0u) {
          owner.cache_regions[bank] = owner.arena->bank_regions[bank];
        }
        owner.bank_regions[bank] = owner.cache_regions[bank];
        owner.bank_regions[bank].count = layout.frame_capacity;
      }
    }
    const auto input = std::find_if(
        pool->graph_owners.begin(), pool->graph_owners.end(),
        [](const PoolPhysicalOwner &owner) {
          return owner.physical_class.role == FrameRole::Input;
        });
    const auto intermediate = std::find_if(
        pool->graph_owners.begin(), pool->graph_owners.end(),
        [](const PoolPhysicalOwner &owner) {
          return owner.physical_class.role == FrameRole::Intermediate;
        });
    const auto output = std::find_if(
        pool->graph_owners.begin(), pool->graph_owners.end(),
        [](const PoolPhysicalOwner &owner) {
          return owner.physical_class.role == FrameRole::Output;
        });
    if (input == pool->graph_owners.end() ||
        intermediate == pool->graph_owners.end() ||
        output == pool->graph_owners.end()) {
      std::terminate();
    }
    pool->input_regions = input->bank_regions;
    pool->first_intermediate_frame = intermediate->bank_regions[0].first;
    pool->intermediate_frame_count = layout.frame_capacity * Pool::BankCount;
    pool->first_output_frame = output->bank_regions[0].first;
    pool->output_frame_count = layout.frame_capacity * Pool::BankCount;
    pool->first_host_input_frame = first_host_input_frame;
    pool->host_input_frame_count = host_input_frame_count;
    pool->first_host_output_frame = first_host_output_frame;
    pool->host_output_frame_count = host_output_frame_count;
    transaction.publish(pool, std::move(pool_admission), host_storage_bytes);
    return pool;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

} // namespace rund::compute::detail::residency::acquire_detail
