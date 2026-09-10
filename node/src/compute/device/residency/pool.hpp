#pragma once

#include "executor.hpp"
#include "persist.hpp"
#include "prefetch.hpp"
#include "registry.hpp"

#include <rund/compute/abi/state.hpp>
#include <rund/compute/fixed.hpp>
#include <rund/storage.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund::compute::detail {
struct BufferState;
struct DeviceState;

namespace residency {

struct TiledGraphPhysicalClass;
namespace acquire_detail {
class Transaction;
}

struct PoolLayout final {
  Type input_type{Type::I32};
  FixedFormat input_format{};
  Type intermediate_type{Type::I32};
  FixedFormat intermediate_format{};
  Type control_type{Type::U64};
  FixedFormat control_format{};
  Type output_type{Type::I32};
  FixedFormat output_format{};
  std::uint64_t input_page_bytes{};
  // Zero means a direct Input->Output transform. For Graph pipelines this is
  // only the named terminal/special-stage compatibility geometry; the
  // planner's physical-class table and graph_pool_footprint() own the complete
  // set of colored Intermediate materializations.
  std::uint64_t intermediate_page_bytes{};
  // Per-frame active-count control consumed by prepared graph stages. It is
  // ordinary immutable execution control, not a page/cache authority.
  std::uint64_t control_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint32_t frame_capacity{};
  // Number of disjoint Graph external-input Host rings retained per bank.
  // Physical order is [bank][input][slot]. Direct execution keeps the default
  // unary group; Graph preparation seals the exact public input count.
  std::uint32_t graph_host_input_count{1u};
  std::uint32_t host_frame_capacity{};
  // Per-bank Host output staging ring. It is independent of the K-sized
  // Device output bank so Drain may release Device storage before Persist.
  std::uint32_t host_output_frame_capacity{};
  // Graph Host Forecast/Promote owns graph_host_input_count input rings. A
  // false value is valid only for a DeviceVsm-required, whole-run resident
  // Graph route; the Pool still owns exact Device physical classes but its
  // Host rings are never an execution authority for those inputs.
  bool graph_host_service{true};

  [[nodiscard]] constexpr bool
  operator==(const PoolLayout &) const noexcept = default;
};

// Checked physical footprint shared by preparation and Pool acquisition.
// Host rings are a separate allocation from native execution arenas; this
// projection owns only their byte accounting and never chooses a route.
struct PoolFootprint final {
  std::uint64_t host_input_bytes{};
  std::uint64_t host_output_bytes{};
  std::uint64_t host_storage_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const PoolFootprint &) const noexcept = default;
};

[[nodiscard]] bool project_pool_footprint(
    const PoolLayout &, Backend, PoolFootprint &) noexcept;

class Pool;
class PhysicalArena;

// One Registry is constructed with one Device and owns the only mutable VSM
// mapping/replacement state and execution gate. Pools are physical frame-class
// owners only; they cannot instantiate a competing policy authority.
class Registry final {
public:
  [[nodiscard]] std::shared_ptr<Pool>
  acquire(const std::shared_ptr<DeviceState> &device, const PoolLayout &layout,
          std::span<const TiledGraphPhysicalClass> graph_classes = {}) noexcept;

  [[nodiscard]] Authority &authority() noexcept { return authority_; }
  [[nodiscard]] const Authority &authority() const noexcept {
    return authority_;
  }
  [[nodiscard]] std::mutex &execution_gate() noexcept {
    return execution_gate_;
  }

private:
  friend class Pool;
  friend class PhysicalArena;
  friend class acquire_detail::Transaction;
  [[nodiscard]] bool release(Pool &pool) noexcept;
  [[nodiscard]] bool release(PhysicalArena &arena) noexcept;

  std::mutex gate_;
  std::mutex execution_gate_;
  Authority authority_;
  std::vector<std::weak_ptr<Pool>> pools_;
  // Pools retain compatible physical classes while live; the Registry keeps
  // only discovery references and never creates a Device ownership cycle.
  std::vector<std::weak_ptr<PhysicalArena>> arenas_;
  std::uint64_t next_extent_{1u};
  std::uint64_t next_view_{1u};
};

struct PhysicalClass final {
  Type type{Type::I32};
  FixedFormat format{};
  std::uint64_t page_bytes{};
  FrameTier tier{FrameTier::Device};
  FrameRole role{FrameRole::Input};

  [[nodiscard]] constexpr bool
  operator==(const PhysicalClass &) const noexcept = default;
};

// Registry-owned stable physical byte generation. `physical_class` is the
// canonical allocation descriptor chosen by the first Pool; later Graph Pools
// in the same tier and committed per-bank storage bin may retain typed
// BufferState views over these same two byte owners. Incompatible semantic
// geometry receives distinct Authority view regions over those bytes.
// Role/Type/FixedFormat/page geometry remain semantic view/cache identity,
// never a second byte allocation or mapping authority.
class PhysicalArena final {
public:
  ~PhysicalArena();
  PhysicalClass physical_class{};
  Registry *registry{};
  std::array<std::shared_ptr<BufferState>, 2u> buffers;
  std::array<FrameRegion, 2u> bank_regions{};
  std::uint64_t extent{};
  std::uint64_t view{};
  std::uint64_t bank_bytes{};
  std::uint32_t frame_capacity{};
  bool host_visible_preferred{};
  storage::Reservation memory;
};

struct PoolPhysicalOwner final {
  std::uint32_t physical_id{};
  std::uint32_t color{};
  // Semantic class of this Pool's typed view. It may differ in Type/format
  // from arena->physical_class while sharing its exact raw allocation.
  PhysicalClass physical_class{};
  std::shared_ptr<PhysicalArena> arena;
  std::array<std::shared_ptr<BufferState>, 2u> buffers;
  // Complete semantic view coordinates over the shared raw banks. They equal
  // arena->bank_regions for the canonical geometry; another page geometry or
  // role owns distinct Authority metadata rows over the same bytes.
  std::array<FrameRegion, 2u> cache_regions{};
  std::array<FrameRegion, 2u> bank_regions{};
  std::uint64_t view{};
  std::uint32_t frame_capacity{};
  bool owns_regions{};
};

class Pool final {
public:
  static constexpr std::uint32_t BankCount = 2u;

  ~Pool();
  PoolLayout layout{};
  std::shared_ptr<DeviceState> device;
  Registry *registry{};
  // Each Pool borrows a K-prefix of two canonical arena banks whose per-bank
  // capacity C may be larger. Runtime coordinates must consume these explicit
  // regions and must never reconstruct bank 1 as first + K.
  std::array<FrameRegion, BankCount> input_regions{};
  std::uint32_t first_intermediate_frame{};
  std::uint32_t intermediate_frame_count{};
  std::uint32_t first_output_frame{};
  std::uint32_t output_frame_count{};
  std::uint32_t first_host_input_frame{};
  std::uint32_t host_input_frame_count{};
  std::uint32_t first_host_output_frame{};
  std::uint32_t host_output_frame_count{};
  std::shared_ptr<PhysicalArena> input_arena;
  // The two canonical backend-local banks are also the prepared execution
  // storage (CPU host buffers or accelerator device buffers). The CPU form is
  // not a separately capacity-managed Host-cache tier. There is no
  // cache/execution mirror or second synchronization authority.
  std::array<std::shared_ptr<BufferState>, BankCount> input;
  std::array<std::shared_ptr<BufferState>, BankCount> intermediate;
  std::array<std::shared_ptr<BufferState>, BankCount> control;
  std::array<std::shared_ptr<BufferState>, BankCount> output;
  // Canonical Graph physical-owner table in planner physical_id order. The
  // legacy named views above remain only for the Direct/special transfer
  // helpers; Graph binding and execution resolve owners through this table.
  std::vector<PoolPhysicalOwner> graph_owners;
  // Accelerator Host input and output arenas are physical VSM frames. Both
  // are registered regions in the same Authority; this object owns bytes,
  // never a second mapping, replacement, or dirty-page authority.
  std::unique_ptr<std::byte[]> host_storage;
  std::uint64_t host_storage_bytes{};
  // CPU retains exactly one cold worker. Accelerator callback receipts live in
  // an accelerator-only allocation, so the CPU object and hot worker carry no
  // native async transaction state.
  Executor executor;
  std::unique_ptr<AcceleratorExecutionRing> accelerator_execution;
  // Two fixed workers let a persistent backing fill the authoritative Host
  // frame banks for e+1 and e+2. Workers own request metadata, never payload.
  std::array<Prefetcher, 2u> prefetch;
  // Accelerator Graph output persistence is independently callback-return
  // gated from Device Drain. The allocation is absent from CPU and Direct
  // pools, preserving their hot object and worker closure.
  std::unique_ptr<GraphPersistRing> graph_persist;
  // The global pool is charged to the Device's sole Pipeline budget for its
  // complete lifetime; Pipeline-local plans only retain a shared reference.
  storage::Reservation memory;
  std::uint64_t host_bytes{};
  bool host_accounted{};

  [[nodiscard]] Authority &authority() noexcept {
    return registry->authority();
  }
  [[nodiscard]] const Authority &authority() const noexcept {
    return registry->authority();
  }
  [[nodiscard]] bool configure_execution() noexcept;
  [[nodiscard]] bool
  submit_execution(std::uint32_t bank,
                   const std::shared_ptr<PipelineState> &pipeline,
                   EpochLease lease, bool defer_generation = false,
                   std::uint64_t control_generation =
                       std::numeric_limits<std::uint64_t>::max(),
                   std::uint8_t control_parity = 0u,
                   std::uint64_t publication_generation =
                       std::numeric_limits<std::uint64_t>::max(),
                   std::uint8_t publication_parity = 0u) noexcept;
  [[nodiscard]] ExecutionReceipt wait_execution(std::uint32_t bank) noexcept;
  [[nodiscard]] bool submit_recurrent(AcceleratorServiceTask task,
                                      void *user) noexcept;
  void wait_recurrent() noexcept;
  [[nodiscard]] std::mutex &execution_gate() noexcept {
    return registry->execution_gate();
  }
  [[nodiscard]] const PoolPhysicalOwner *
  graph_owner(std::uint32_t physical_id) const noexcept;
  // Exact logical Host/Metadata extent retained by this Pool: the Pool object,
  // source-owned vector capacities, unique arena and BufferState objects, and
  // metadata-only prefetch arrays. Physical Buffer/Host frame payload remains
  // in the existing resident/host/device producers.
  [[nodiscard]] std::uint64_t retained_host_bytes() const noexcept;
};

// Validates that the Pool retains the planner's exact physical-class table and
// returns its unique per-bank VSM footprint. Pipeline admission and memory
// projection share this producer instead of reconstructing named role bytes.
[[nodiscard]] bool
graph_pool_footprint(const Pool &pool,
                     std::span<const TiledGraphPhysicalClass> classes,
                     std::uint64_t &bytes) noexcept;

} // namespace residency
} // namespace rund::compute::detail
