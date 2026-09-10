#pragma once

#include "../../pipeline/residency/model.hpp"
#include "cycle/plan.hpp"
#include "execution/evidence.hpp"
#include "execution/graph_forecast.hpp"
#include "execution/graph_persist/capacity.hpp"
#include "execution/registration/state.hpp"
#include "registry/cache_model.hpp"
#include "registry/credentials.hpp"
#include "registry/cycle_owner.hpp"
#include "registry/graph.hpp"
#include "registry/model/cpu.hpp"
#include "registry/model/cycle.hpp"
#include "registry/model/execution.hpp"
#include "registry/model/frame.hpp"
#include "registry/model/lease.hpp"
#include "registry/model/state.hpp"
#include "registry/model/view.hpp"
#include "registry/transfer.hpp"
#include "registry/view_receipt.hpp"

#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund::compute::detail::graph_reduce {
struct CpuGraphQuarantine;
class CpuReceiptBook;
} // namespace rund::compute::detail::graph_reduce

namespace rund::compute::detail::residency {

class DirectRecurrenceRegistration;
class DeviceVsmRegistration;

class Authority;
class VirtualTransactionOwner;
class GraphForecastOwner;
class GraphPromoteOwner;
class GraphDrainOwner;
class GraphPersistOwner;
class SlidingOwner;
class DirectRecurrenceOwner;
class ExecutionOwner;
class CpuGraphOwner;
class ViewOwner;
class ResidencyPlan;

namespace graph_epoch_detail {
struct AdmissionDraft;
class Validation;
class Assignment;
class Relocation;
} // namespace graph_epoch_detail

namespace execution {
class Plan;
class Sliding;
class SlidingFinal;
class SlidingFetch;
class SlidingPromote;
class SlidingNative;
class SlidingDrain;
class SlidingPersist;
class GraphForecast;
class GraphReady;
class GraphPromote;
class GraphDrain;
class GraphPersist;
struct GraphPersistPage;
struct GraphForecastCompletion;
struct GraphPromoteCompletion;
struct GraphDrainCompletion;
struct GraphPersistCompletion;
struct SlidingProjection;
struct SlidingEvidence;
struct Node;
} // namespace execution

namespace release_detail {
struct ReleaseCheck;
} // namespace release_detail

class Authority final {
public:
  using Frame = registry_model::Frame;
  using ViewCommitReceipt = registry_model::ViewCommitReceipt;
  using LeaseState = registry_model::LeaseState;
  using LeaseSlot = registry_model::LeaseSlot;

  Authority() noexcept;
  ~Authority() noexcept;
  Authority(const Authority &) = delete;
  Authority &operator=(const Authority &) = delete;

  // Physical frame and resident-view registration.
  [[nodiscard]] bool register_frames(FrameTier tier, FrameRole role,
                                     std::uint32_t frame_capacity,
                                     std::uint32_t &first_frame) noexcept;
  [[nodiscard]] bool register_frames(FrameTier tier, FrameRole role,
                                     std::uint32_t frame_capacity,
                                     std::uint64_t extent, std::uint64_t view,
                                     std::uint32_t &first_frame) noexcept;
  [[nodiscard]] bool
  register_resident_recurrence_view(FrameRole, ResidentRecurrenceView,
                                    ResidentRecurrenceBinding &) noexcept;
  [[nodiscard]] bool
  release_frames(std::span<const FrameRegion> regions) noexcept;
  [[nodiscard]] bool
  owns_regions(std::span<const FrameRegion> regions) const noexcept;
  // Transactional view selection and receipt disposition.
  [[nodiscard]] ViewOwner views() noexcept;
  // Epoch, transform, and Graph admission.
  [[nodiscard]] AuthorityResult begin(std::span<const CacheUse> uses,
                                      FrameTier tier, FrameRole role,
                                      std::uint32_t first_frame,
                                      std::uint32_t frame_count,
                                      CpuReservationKey cpu_key = {}) noexcept;
  [[nodiscard]] bool issue_alias(
      std::uint64_t owner_token, std::span<const FrameRegion> source_regions,
      FrameRegion target_region, CacheKey source_key, CacheKey target_key,
      std::uint32_t target_frame, std::uint64_t source_offset,
      std::uint64_t target_offset, std::uint64_t bytes,
      std::uint64_t frame_bytes, AliasLease &) noexcept;
  [[nodiscard]] bool release_alias(AliasLease &&, bool source_known) noexcept;
  [[nodiscard]] AuthorityResult
  begin_transform(std::span<const CacheUse> inputs, FrameRegion input_region,
                  std::span<const CacheUse> outputs,
                  FrameRegion output_region) noexcept;
  [[nodiscard]] AuthorityResult
  begin_graph_transform(std::span<const PageUse> inputs,
                        GraphMaterialization input, FrameRegion input_region,
                        std::span<const PageUse> outputs,
                        GraphMaterialization output, FrameRegion output_region,
                        std::uint64_t epoch) noexcept;
  [[nodiscard]] AuthorityResult
  begin_graph(std::span<const PageUse> uses,
              GraphMaterialization materialization, FrameRegion region,
              std::uint64_t epoch) noexcept;
  // Stateless lifecycle facets borrow this Authority's sole mutable storage.
  [[nodiscard]] GraphForecastOwner graph_forecasts() noexcept;
  [[nodiscard]] GraphPromoteOwner graph_promotes() noexcept;
  [[nodiscard]] GraphDrainOwner graph_drains() noexcept;
  [[nodiscard]] GraphPersistOwner graph_persists() noexcept;
  [[nodiscard]] AuthorityResult
  begin_graph_epoch(std::span<const PageUse> uses,
                    std::span<const GraphPortRequest> ports,
                    std::size_t anchor_port, std::uint64_t epoch,
                    CpuReservationKey cpu_key = {}) noexcept;
  [[nodiscard]] bool activate(std::uint64_t token) noexcept;
  [[nodiscard]] AuthorityResult resume(std::uint64_t token) noexcept;
  // Recurrent execution admission and terminal protocol.
  [[nodiscard]] ExecutionOwner executions() noexcept;
  [[nodiscard]] SlidingOwner sliding() noexcept;
  [[nodiscard]] DirectRecurrenceOwner direct_recurrences() noexcept;
  // Drain and rolling-cycle admission.
  [[nodiscard]] CycleOwner cycles() noexcept;
  [[nodiscard]] AuthorityResult
  begin_writeback(std::span<const CacheKey> keys, std::uint32_t first_frame,
                  std::uint32_t frame_count) noexcept;
  [[nodiscard]] AuthorityResult
  begin_migration(std::span<const CacheKey> keys, std::uint32_t first_frame,
                  std::uint32_t frame_count) noexcept;
  [[nodiscard]] AuthorityResult
  begin_discard(std::span<const CacheKey> keys, std::uint32_t first_frame,
                std::uint32_t frame_count) noexcept;
  [[nodiscard]] AuthorityResult drain_dirty(std::uint32_t first_frame,
                                            std::uint32_t frame_count) noexcept;
  [[nodiscard]] bool complete(std::uint64_t token, bool success,
                              bool invalidate_all = false) noexcept;
  // CPU Graph lifecycle borrows this Authority's sole gate and state.
  [[nodiscard]] CpuGraphOwner cpu_graph() noexcept;
  [[nodiscard]] bool
  complete_migration(std::uint64_t source_token,
                     std::uint64_t destination_token) noexcept;
  [[nodiscard]] bool discard(std::uint64_t token) noexcept;
  [[nodiscard]] bool probe(std::span<const CacheKey> keys,
                           std::span<std::uint8_t> resident,
                           std::uint32_t first_frame,
                           std::uint32_t frame_count) const noexcept;
  // Virtual backing and frozen schedule ownership.
  [[nodiscard]] VirtualTransactionOwner virtual_transactions() noexcept;

  [[nodiscard]] std::uint32_t frame_capacity() const noexcept;

private:
  friend class CycleOwner;
  friend class VirtualTransactionOwner;
  friend class GraphForecastOwner;
  friend class GraphPromoteOwner;
  friend class GraphDrainOwner;
  friend class GraphPersistOwner;
  friend class SlidingOwner;
  friend class DirectRecurrenceOwner;
  friend class ExecutionOwner;
  friend class CpuGraphOwner;
  friend class ViewOwner;
  friend class graph_epoch_detail::Validation;
  friend class graph_epoch_detail::Assignment;
  friend class graph_epoch_detail::Relocation;
  friend struct registry_model::PendingCpu;
  friend struct registry_model::RetryTxn;
  friend class registry_model::CpuQuarantineOwner;
  [[nodiscard]] registry_model::ViewCommitState
  view_commit_state_locked() const noexcept;
  [[nodiscard]] bool view_commit_inflight_locked() const noexcept;
  [[nodiscard]] bool view_commit_quarantined_locked() const noexcept;
  [[nodiscard]] bool
  view_commit_active_locked(const ViewCommitReceipt &) const noexcept;
  [[nodiscard]] bool close_rows_clear_locked() const noexcept;
  [[nodiscard]] AuthorityResult begin_graph_epoch_locked(
      std::span<const PageUse>, std::span<const GraphPortRequest>,
      std::size_t anchor_port, std::uint64_t epoch, CpuReservationKey,
      const GraphPersistIdentity *retry_identity) noexcept;
  friend class DirectRecurrenceRegistration;
  friend class DeviceVsmRegistration;
  [[nodiscard]] RegistrationResult rollback_direct_registration(
      std::span<const ResidentRecurrenceBinding>) noexcept;
  friend struct release_detail::ReleaseCheck;
  [[nodiscard]] ExecutionLease admit_execution(const execution::Plan &plan,
                                               bool window,
                                               bool sliding = false) noexcept;
  [[nodiscard]] bool
  validate_execution_regions_locked(const execution::Plan &, bool sliding,
                                    registry_model::ExecutionSlot &) noexcept;
  [[nodiscard]] AuthorityFailure
  admit_direct_execution_locked(const execution::Plan &,
                                registry_model::ExecutionSlot &) noexcept;
  [[nodiscard]] AuthorityResult begin_drain(std::span<const CacheKey> keys,
                                            std::uint32_t first_frame,
                                            std::uint32_t frame_count,
                                            TransitionKind kind) noexcept;
  [[nodiscard]] AuthorityResult begin_locked(std::span<const CacheUse>,
                                             FrameTier, FrameRole,
                                             std::uint32_t, std::uint32_t,
                                             CpuReservationKey) noexcept;
  [[nodiscard]] AuthorityResult begin_graph_locked(std::span<const PageUse>,
                                                   GraphMaterialization,
                                                   FrameRegion,
                                                   std::uint64_t) noexcept;
  [[nodiscard]] bool activate_locked(std::uint64_t) noexcept;
  [[nodiscard]] AuthorityResult resume_locked(std::uint64_t) noexcept;
  [[nodiscard]] AuthorityResult begin_drain_locked(std::span<const CacheKey>,
                                                   std::uint32_t, std::uint32_t,
                                                   TransitionKind) noexcept;
  [[nodiscard]] bool
  complete_locked(std::uint64_t, bool, bool,
                  bool allow_forecast_quarantine = false) noexcept;
  [[nodiscard]] bool register_frames_impl(FrameTier, FrameRole, std::uint32_t,
                                          std::uint64_t, std::uint64_t,
                                          FrameState, CacheKey,
                                          std::uint32_t &) noexcept;
  [[nodiscard]] bool ensure_view_commit_storage_locked(std::size_t) noexcept;
  [[nodiscard]] bool graph_forecast_quarantine_active_locked() const noexcept;
  [[nodiscard]] bool graph_forecast_quarantine_slot_locked() const noexcept;
  mutable std::mutex gate_;
  std::vector<Frame> frames_;
  // The gate and physical frame table remain the façade's synchronization and
  // storage boundary. Each component below owns one mutable authority domain;
  // no field is mirrored outside its component.
  registry_model::ExecutionAuthorityState execution_state_{};
  registry_model::CycleAuthorityState cycle_state_{};
  registry_model::CpuGraphAuthorityState cpu_graph_state_{};
  registry_model::ViewAuthorityState view_state_{};
  registry_model::CredentialsState credentials_{};
};

} // namespace rund::compute::detail::residency
