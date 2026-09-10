#pragma once

#include "../../../../pipeline/residency/model.hpp"
#include "../../execution/evidence.hpp"
#include "../../execution/registration/state.hpp"
#include "../credentials/direct.hpp"
#include "../credentials/execution.hpp"
#include "../credentials/sliding.hpp"
#include "../transfer.hpp"
#include "frame.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail::residency::registry_model {

inline constexpr std::size_t ExecutionRegionCapacity = 8u;
inline constexpr std::size_t ExecutionFrameCapacity =
    ExecutionRegionCapacity * 32u;
inline constexpr std::size_t ExecutionBindingCapacity = 2u * 32u;
inline constexpr std::size_t ExecutionTransitionCapacity = 5u * 32u;

// The fixed execution journal is one physical record, not a union of
// ordinary and Sliding sub-journals.  Authority composes exactly one of it;
// this header owns only its layout and capacities.
struct ExecutionSlot final {
  // Present only for the service-free aggregate path. Known close clears it;
  // Unknown deliberately retains it with pinned physical rows.
  std::shared_ptr<const void> direct_proof_owner{};
  std::shared_ptr<const registration_detail::State> registration_state{};
  std::uint64_t registration_nonce{};
  std::array<ResidentRecurrenceBinding, ExecutionBindingCapacity>
      direct_bindings{};
  std::size_t direct_binding_count{};
  bool direct_released{};
  bool direct_success{};
  std::array<std::array<std::uint64_t, 3u>, 2u> issued{};
  std::array<std::array<std::uint64_t, 3u>, 2u> terminals{};
  std::array<std::array<std::uint64_t, 3u>, 2u> sequences{};
  std::array<std::array<bool, 3u>, 2u> may_write{};
  std::array<std::uint32_t, ExecutionFrameCapacity> frames{};
  // Raw Sliding retention is authenticated by the exact key handed off by
  // this generation. These fixed rows are indexed by `frames`, never by Q.
  std::array<CacheKey, ExecutionFrameCapacity> observed_frame_keys{};
  std::array<bool, ExecutionFrameCapacity> observed_frame_valid{};
  std::array<std::uint32_t, ExecutionFrameCapacity> undo_frames{};
  std::array<Frame, ExecutionFrameCapacity> undo{};
  std::array<std::array<CacheBinding, ExecutionBindingCapacity>, 2u>
      service_bindings{};
  std::array<std::array<CacheTransition, ExecutionTransitionCapacity>, 2u>
      service_transitions{};
  std::array<std::size_t, 2u> service_binding_count{};
  std::array<std::size_t, 2u> service_transition_count{};
  std::array<std::uint32_t, 2u> service_backing_mask{};
  std::array<std::uint32_t, 2u> service_transfer_mask{};
  // Q1 keeps the exact full-run service ticket above. A bounded native
  // window instead reuses one fixed ticket per {Input|Output, bank}; an
  // epoch may overwrite its bank slot only after that slot's exact terminal
  // and native release made the physical owner reusable.
  std::array<std::array<std::array<CacheBinding, ExecutionBindingCapacity>, 2u>,
             2u>
      window_service_bindings{};
  std::array<
      std::array<std::array<CacheTransition, ExecutionTransitionCapacity>, 2u>,
      2u>
      window_service_transitions{};
  std::array<std::array<std::size_t, 2u>, 2u> window_service_binding_count{};
  std::array<std::array<std::size_t, 2u>, 2u> window_service_transition_count{};
  std::array<std::array<std::uint32_t, 2u>, 2u> window_service_backing_mask{};
  std::array<std::array<std::uint32_t, 2u>, 2u> window_service_transfer_mask{};
  std::array<std::array<std::uint32_t, 2u>, 2u> window_service_coherent_mask{};
  std::array<std::array<std::uint64_t, 2u>, 2u> window_service_epoch{
      {{NeverUse, NeverUse}, {NeverUse, NeverUse}}};
  // The physical Host/Input layout is sealed at admission. Sliding fetch
  // and promote authenticate every key against this layout before making a
  // row Resident; retaining it at close therefore preserves the exact
  // plan-projected frame view without a Q-sized cache mirror.
  std::array<FrameRegion, 2u> sliding_host_input_regions{};
  std::array<ExecutionFailure, ExecutionClose::FailureCapacity> failures{};
  execution::NativeEvidence native{};
  std::array<execution::Release, execution::WindowCapacity> releases{};
  std::array<std::uint64_t, execution::WindowCapacity> release_epochs{
      NeverUse, NeverUse, NeverUse, NeverUse};
  ExecutionProgress progress{};
  std::size_t frame_count{};
  std::size_t undo_count{};
  std::size_t failure_count{};
  std::size_t release_count{};
  std::size_t window_accept_count{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner_nonce{};
  std::uint64_t plan{};
  std::uint64_t epochs{};
  std::uint64_t next_sequence{1u};
  // Tagged by sliding_admitted. Ordinary execution counts native in-flight
  // work; Sliding binds its unforgeable State owner in this same word. One
  // object, rather than two union members, keeps the lifetime well-defined
  // while preserving the CPU Authority layout.
  std::uint64_t native_inflight{};
  bool failed{};
  bool unknown{};
  bool native_accepted{};
  bool native_rejected{};
  bool cache_admitted{};
  bool window_cache_admitted{};
  bool sliding_admitted{};
  bool direct_recurrence_admitted{};
  bool output_admitted{};
  bool window_final{};
  SlidingFinalPhase sliding_final_phase{SlidingFinalPhase::None};
};

} // namespace rund::compute::detail::residency::registry_model
