#pragma once

#include "../../pool.hpp"

#include "../../../../buffer/local.hpp"
#include "../../../state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail::residency::acquire_detail {

// Strong discovery references are held for the complete duration of the
// Registry gate.  A weak entry may otherwise become the last owner while a
// transaction is still constructing a Pool and re-enter the same gate.
struct RegistryKeepalive final {
  std::vector<std::shared_ptr<Pool>> pools;
  std::vector<std::shared_ptr<PhysicalArena>> arenas;
};

enum class ArenaReuse : std::uint8_t { GraphExtent, OrdinaryInput };
enum class ArenaRegistration : std::uint8_t { GraphContiguous, OrdinaryBanks };
enum class BufferKind : std::uint8_t { InputBinding, Residency };

// One physical allocation admission.  Graph and ordinary projections both
// use this record; only their semantic wiring and region projection differ.
struct PendingPhysicalArena final {
  // Declared before the arena so construction failure destroys native buffers
  // before refunding their reservation.
  storage::Reservation admission;
  std::shared_ptr<PhysicalArena> arena;
  std::uint64_t committed_bytes{};
  ArenaRegistration registration{ArenaRegistration::GraphContiguous};
  bool fresh{};
};

// Shared physical reservation/arena/buffer/frame-registration transaction.
// Registry::acquire owns the public gate; this object owns every fallible
// operation between that gate and publication.  Its destructor is the single
// rollback boundary for registered frame regions and admitted bytes.
class Transaction final {
public:
  Transaction(Registry &, const std::shared_ptr<DeviceState> &,
              const PoolLayout &, RegistryKeepalive &) noexcept;
  ~Transaction();

  Transaction(const Transaction &) = delete;
  Transaction &operator=(const Transaction &) = delete;

  [[nodiscard]] bool reserve_registry(std::size_t additional_arenas) noexcept;
  [[nodiscard]] bool reserve_pending(std::size_t count) noexcept;

  // Finds the best live compatible arena or creates one physical arena and
  // its two backend buffers.  The request is appended to pending() exactly
  // once, so no projection can duplicate arena accounting or rollback.
  [[nodiscard]] bool prepare_arena(const PhysicalClass &,
                                   std::uint32_t frame_capacity,
                                   bool host_visible_preferred, ArenaReuse,
                                   ArenaRegistration, std::size_t &index);

  [[nodiscard]] PendingPhysicalArena &pending(std::size_t index) noexcept {
    return pending_[index];
  }
  [[nodiscard]] const PendingPhysicalArena &
  pending(std::size_t index) const noexcept {
    return pending_[index];
  }
  [[nodiscard]] std::uint64_t next_view_id() noexcept;

  [[nodiscard]] bool reserve_budget(std::uint64_t bytes,
                                    storage::Reservation &) const noexcept;
  [[nodiscard]] bool make_bank_buffers(
      std::array<std::shared_ptr<BufferState>, Pool::BankCount> &, Type,
      std::size_t count, std::uint64_t physical_bytes, BufferKind);
  [[nodiscard]] bool make_host_storage(Pool &, std::uint64_t bytes);

  // Registers fresh physical arenas using the projection's required layout:
  // Graph uses one extent/view region split into explicit banks; ordinary
  // input uses two independent zero-identity bank regions.
  [[nodiscard]] bool register_new_arenas();
  [[nodiscard]] bool register_region(FrameTier, FrameRole, std::uint32_t,
                                     std::uint64_t extent, std::uint64_t view,
                                     std::uint32_t &first) noexcept;
  [[nodiscard]] bool rollback_regions() noexcept;

  [[nodiscard]] bool commit_pool(storage::Reservation &,
                                 std::uint64_t bytes) noexcept;
  [[nodiscard]] bool commit_arenas() noexcept;

  // Publication is allocation-free after reserve_registry().  It transfers
  // all committed reservations, accounts Host bytes once, and publishes one
  // weak Registry entry for the Pool and each newly-created arena.
  void publish(const std::shared_ptr<Pool> &,
               storage::Reservation &&pool_admission,
               std::uint64_t host_bytes) noexcept;

private:
  Registry &registry_;
  const std::shared_ptr<DeviceState> &device_;
  RegistryKeepalive &keepalive_;
  std::vector<PendingPhysicalArena> pending_;
  std::array<FrameRegion,
             TiledGraphResourceCapacity * Pool::BankCount + 8u>
      registered_{};
  std::size_t registered_count_{};
  bool published_{};
};

[[nodiscard]] std::shared_ptr<Pool>
acquire_graph(Registry &, const std::shared_ptr<DeviceState> &,
              const PoolLayout &, std::span<const TiledGraphPhysicalClass>,
              RegistryKeepalive &) noexcept;

[[nodiscard]] std::shared_ptr<Pool>
acquire_ordinary(Registry &, const std::shared_ptr<DeviceState> &,
                 const PoolLayout &, RegistryKeepalive &) noexcept;

} // namespace rund::compute::detail::residency::acquire_detail
