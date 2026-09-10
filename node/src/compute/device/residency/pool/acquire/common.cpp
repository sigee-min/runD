#include "../../../../buffer/state.hpp"
#include "common.hpp"

#include "../internal.hpp"

#include "../../../../pipeline/residency/model.hpp"
#include "../../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::residency::acquire_detail {

namespace {

[[nodiscard]] bool
extent_compatible(const PhysicalArena &arena, const PhysicalClass &requested,
                  const std::uint32_t frame_capacity,
                  const std::uint64_t committed_bytes,
                  const bool host_visible_preferred) noexcept {
  if (arena.extent == 0u || arena.view == 0u ||
      requested.page_bytes == 0u || arena.bank_bytes == 0u ||
      arena.bank_bytes % requested.page_bytes != 0u ||
      arena.bank_bytes / requested.page_bytes < frame_capacity ||
      arena.physical_class.tier != requested.tier ||
      arena.host_visible_preferred != host_visible_preferred) {
    return false;
  }
  // Extent-zero ordinary arenas support exact-region borrowing only. Their
  // live rows cannot acquire a new semantic view identity through this loan.
  // Type belongs to the admitted BufferState view, not the raw byte extent.
  // acquire_graph creates that typed view while retaining this allocation's
  // sole owner and charge; tier and native memory intent must still match.
  return std::all_of(
      arena.buffers.begin(), arena.buffers.end(),
      [committed_bytes](const std::shared_ptr<BufferState> &buffer) {
        return buffer != nullptr && buffer->bytes != 0u &&
               buffer->physical_bytes == committed_bytes;
      });
}

[[nodiscard]] std::uint64_t next_identity(std::uint64_t &next) noexcept {
  if (next == 0u || next == std::numeric_limits<std::uint64_t>::max()) {
    return 0u;
  }
  return next++;
}

} // namespace

Transaction::Transaction(Registry &registry,
                         const std::shared_ptr<DeviceState> &device,
                         const PoolLayout &,
                         RegistryKeepalive &keepalive) noexcept
    : registry_(registry), device_(device), keepalive_(keepalive) {}

Transaction::~Transaction() {
  if (!published_ && registered_count_ != 0u && !rollback_regions()) {
    std::terminate();
  }
}

bool Transaction::reserve_registry(const std::size_t additional_arenas) noexcept {
  try {
    if (additional_arenas >
            std::numeric_limits<std::size_t>::max() -
                registry_.arenas_.size() ||
        registry_.pools_.size() == std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    registry_.pools_.reserve(registry_.pools_.size() + 1u);
    registry_.arenas_.reserve(registry_.arenas_.size() + additional_arenas);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  } catch (const std::length_error &) {
    return false;
  }
}

bool Transaction::reserve_pending(const std::size_t count) noexcept {
  try {
    pending_.reserve(count);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  } catch (const std::length_error &) {
    return false;
  }
}

bool Transaction::prepare_arena(const PhysicalClass &requested,
                                const std::uint32_t frame_capacity,
                                const bool host_visible_preferred,
                                const ArenaReuse reuse,
                                const ArenaRegistration registration,
                                std::size_t &index) {
  std::shared_ptr<PhysicalArena> selected;
  for (const std::shared_ptr<PhysicalArena> &arena : keepalive_.arenas) {
    if (arena == nullptr) {
      continue;
    }
    if (reuse == ArenaReuse::GraphExtent) {
      const auto candidate_storage =
          arena->bank_bytes != 0u
              ? planned_buffer_storage_bytes(*device_, arena->bank_bytes)
              : Result<std::uint64_t>::fail(Reason::BufferCapacity);
      const bool already_selected =
          std::any_of(pending_.begin(), pending_.end(),
                      [&arena](const PendingPhysicalArena &candidate) {
                        return candidate.arena == arena;
                      });
      // A Pool borrows only its explicit K-prefix. A matching raw extent may
      // lend across semantic page geometry through a separate view region.
      if (!already_selected && candidate_storage &&
          extent_compatible(*arena, requested, frame_capacity,
                            *candidate_storage, host_visible_preferred) &&
          (selected == nullptr || arena->bank_bytes < selected->bank_bytes)) {
        selected = arena;
      }
    } else if (arena->physical_class == requested &&
               arena->host_visible_preferred == host_visible_preferred &&
               arena->frame_capacity >= frame_capacity &&
               (selected == nullptr ||
                arena->frame_capacity < selected->frame_capacity)) {
      selected = arena;
    }
  }

  PendingPhysicalArena candidate{};
  candidate.arena = std::move(selected);
  candidate.registration = registration;
  if (candidate.arena == nullptr) {
    candidate.fresh = true;
    candidate.arena = std::make_shared<PhysicalArena>();
    candidate.arena->physical_class = requested;
    candidate.arena->host_visible_preferred = host_visible_preferred;
    candidate.arena->frame_capacity = frame_capacity;
    if (reuse == ArenaReuse::GraphExtent) {
      candidate.arena->extent = next_identity(registry_.next_extent_);
      candidate.arena->view = next_identity(registry_.next_view_);
      if (candidate.arena->extent == 0u || candidate.arena->view == 0u) {
        return false;
      }
    }

    const std::size_t width = type_bytes(requested.type);
    std::uint64_t arena_bytes = 0u;
    if (width == 0u ||
        !kernel::checked::mul(requested.page_bytes, frame_capacity,
                              arena_bytes) ||
        arena_bytes / width > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    candidate.arena->bank_bytes = arena_bytes;
    const auto storage = planned_buffer_storage_bytes(*device_, arena_bytes);
    if (!storage ||
        !kernel::checked::mul(*storage, Pool::BankCount,
                              candidate.committed_bytes) ||
        !reserve_budget(candidate.committed_bytes, candidate.admission)) {
      return false;
    }
    std::array<std::shared_ptr<BufferState>, Pool::BankCount> buffers{};
    const BufferKind kind = host_visible_preferred
                                ? BufferKind::Residency
                                : BufferKind::InputBinding;
    if (!make_bank_buffers(buffers, requested.type,
                           static_cast<std::size_t>(arena_bytes / width),
                           *storage, kind)) {
      return false;
    }
    candidate.arena->buffers = std::move(buffers);
  }

  index = pending_.size();
  pending_.push_back(std::move(candidate));
  return true;
}

std::uint64_t Transaction::next_view_id() noexcept {
  return next_identity(registry_.next_view_);
}

bool Transaction::reserve_budget(const std::uint64_t bytes,
                                 storage::Reservation &reservation) const
    noexcept {
  reservation = device_->pipeline_memory_budget.reserve(bytes);
  return static_cast<bool>(reservation);
}

bool Transaction::make_bank_buffers(
    std::array<std::shared_ptr<BufferState>, Pool::BankCount> &destination,
    const Type type, const std::size_t count,
    const std::uint64_t physical_bytes, const BufferKind kind) {
  std::array<std::shared_ptr<BufferState>, Pool::BankCount> made{};
  for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
    auto buffer = kind == BufferKind::Residency
                      ? make_planned_residency_buffer(
                            device_, type, count, physical_bytes)
                      : make_planned_input_binding_buffer(
                            device_, type, count, physical_bytes);
    if (!buffer) {
      return false;
    }
    made[bank] = std::move(buffer).value();
  }
  destination = std::move(made);
  return true;
}

bool Transaction::make_host_storage(Pool &pool, const std::uint64_t bytes) {
  if (bytes != 0u) {
    pool.host_storage =
        std::make_unique<std::byte[]>(static_cast<std::size_t>(bytes));
  }
  pool.host_storage_bytes = bytes;
  return true;
}

bool Transaction::register_new_arenas() {
  for (PendingPhysicalArena &candidate : pending_) {
    if (!candidate.fresh) {
      continue;
    }
    PhysicalArena &arena = *candidate.arena;
    const std::uint32_t capacity = arena.frame_capacity;
    if (capacity == 0u ||
        capacity > std::numeric_limits<std::uint32_t>::max() /
                       Pool::BankCount) {
      return false;
    }
    const FrameTier tier = arena.physical_class.tier;
    const FrameRole role = arena.physical_class.role;
    if (candidate.registration == ArenaRegistration::GraphContiguous) {
      std::uint32_t first = 0u;
      if (!register_region(tier, role, capacity * Pool::BankCount,
                           arena.extent, arena.view, first)) {
        return false;
      }
      for (std::uint32_t bank = 0u; bank < Pool::BankCount; ++bank) {
        arena.bank_regions[bank] = FrameRegion{
            .tier = tier,
            .role = role,
            .first = first + bank * capacity,
            .count = capacity,
        };
      }
    } else {
      for (std::uint32_t bank = 0u; bank < Pool::BankCount; ++bank) {
        std::uint32_t first = 0u;
        if (!register_region(tier, role, capacity, 0u, 0u, first)) {
          return false;
        }
        arena.bank_regions[bank] = FrameRegion{
            .tier = tier, .role = role, .first = first, .count = capacity};
      }
    }
  }
  return true;
}

bool Transaction::register_region(const FrameTier tier, const FrameRole role,
                                  const std::uint32_t count,
                                  const std::uint64_t extent,
                                  const std::uint64_t view,
                                  std::uint32_t &first) noexcept {
  if (count == 0u) {
    first = 0u;
    return true;
  }
  if (registered_count_ == registered_.size() ||
      !registry_.authority_.register_frames(tier, role, count, extent, view,
                                             first)) {
    return false;
  }
  registered_[registered_count_++] =
      FrameRegion{.tier = tier, .role = role, .first = first, .count = count};
  return true;
}

bool Transaction::rollback_regions() noexcept {
  if (registered_count_ == 0u) {
    return true;
  }
  const bool released = registry_.authority_.release_frames(
      std::span<const FrameRegion>{registered_.data(), registered_count_});
  if (released) {
    registered_count_ = 0u;
  }
  return released;
}

bool Transaction::commit_pool(storage::Reservation &reservation,
                              const std::uint64_t bytes) noexcept {
  return static_cast<bool>(reservation.commit(
      storage::Usage{.physical_bytes = bytes, .allocated_bytes = bytes}));
}

bool Transaction::commit_arenas() noexcept {
  for (PendingPhysicalArena &candidate : pending_) {
    if (!candidate.fresh) {
      continue;
    }
    if (!static_cast<bool>(candidate.admission.commit(storage::Usage{
            .physical_bytes = candidate.committed_bytes,
            .allocated_bytes = candidate.committed_bytes}))) {
      return false;
    }
  }
  return true;
}

void Transaction::publish(const std::shared_ptr<Pool> &pool,
                          storage::Reservation &&pool_admission,
                          const std::uint64_t host_bytes) noexcept {
  pool->memory = std::move(pool_admission);
  for (PendingPhysicalArena &candidate : pending_) {
    if (!candidate.fresh) {
      continue;
    }
    candidate.arena->memory = std::move(candidate.admission);
    candidate.arena->registry = &registry_;
    registry_.arenas_.emplace_back(candidate.arena);
  }
  pool->host_bytes = host_bytes;
  {
    std::lock_guard memory_lock{device_->memory.gate};
    device_->memory.host.current = ::rund::detail::counter::SaturatingAdd(
        device_->memory.host.current, pool->host_bytes);
    device_->memory.host.peak =
        std::max(device_->memory.host.peak, device_->memory.host.current);
    device_->memory.host.cumulative = ::rund::detail::counter::SaturatingAdd(
        device_->memory.host.cumulative, pool->host_bytes);
  }
  pool->host_accounted = true;
  pool->registry = &registry_;
  registry_.pools_.emplace_back(pool);
  registered_count_ = 0u;
  published_ = true;
}

} // namespace rund::compute::detail::residency::acquire_detail
