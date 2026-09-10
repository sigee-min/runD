#pragma once

#include "residency/authority.hpp"
#include "state.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] Status acquire_claims(DeviceState &device,
                                    std::span<const BufferClaim> claims,
                                    bool reject_poison = true) noexcept;
void release_claims(DeviceState &device,
                    std::span<const BufferClaim> claims) noexcept;
// Publishes write generation or poison and releases the complete claim set in
// one Device claim-gate acquisition. `poison_writes` applies only on failure.
void publish_claims(DeviceState &device, std::span<const BufferClaim> claims,
                    bool succeeded, bool poison_writes) noexcept;
[[nodiscard]] bool buffer_poisoned(const BufferState &buffer) noexcept;
// One device-affinity predicate for every prepare-time Pipeline resource path.
// Callers may order this check at their existing admission boundary, but may
// not reconstruct owner/device equality locally.
[[nodiscard]] Status
validate_pipeline_resource_device(const PipelineState &state,
                                  const PipelineResource &resource) noexcept;
[[nodiscard]] Status
validate_pipeline_resources(const PipelineState &state) noexcept;
[[nodiscard]] Status acquire_pipeline_claims(PipelineState &state) noexcept;
void close_pipeline_observation_epoch(PipelineState &state) noexcept;
void synchronize_pipeline_observation_epoch(
    PipelineState &state, const PipelinePublicationState &publication) noexcept;

struct PipelineTerminal final {
  static constexpr std::uint64_t no_publication_generation =
      std::numeric_limits<std::uint64_t>::max();
  Reason reason{Reason::Ok};
  std::size_t verified{};
  // Zero denotes the complete prepared Pipeline. A nonzero successful prefix
  // is the exact step count issued under an external VSM lease; it may never
  // be widened to the prepared capacity for evidence or owner publication.
  std::size_t issued_steps{};
  std::size_t failed_step{};
  bool failure_step_known{};
  bool writes_possible{};
  // True only when CPU never entered publication or a valid device control
  // proved the success-gated publication shader took its no-store path.
  bool publication_suppressed{};
  // Private aggregate Scan terminals validate against this canonical base but
  // advance only the private native generation until final run commit.
  bool defer_generation{};
  std::uint64_t publication_generation{no_publication_generation};
  std::uint8_t publication_parity{};
};

// Aggregate ordinary Scan commits its fixed two-bank physical cursor through
// this claim-owned critical section. Canonical and observation identities are
// validated against the captured bases before either bank is advanced.
struct DeferredPipelinePublication final {
  std::uint64_t base_generation{};
  std::uint64_t base_payload_epoch{};
  std::uint64_t control_generation{};
  std::uint64_t terminal_count{};
  std::uint8_t base_parity{};
};

[[nodiscard]] Status preflight_deferred_pipeline_generations(
    PipelineState &, DeferredPipelinePublication, PipelineState &,
    DeferredPipelinePublication) noexcept;
// The no-fail half of the deferred commit. The caller must have completed the
// matching preflight while the same run owns both Pipeline states.
void apply_deferred_pipeline_generations(
    PipelineState &, DeferredPipelinePublication, PipelineState &,
    DeferredPipelinePublication) noexcept;

// Publishes the terminal Pipeline state and releases every resource claim.
// Write poison/generation publication happens under the same Device claim-gate
// acquisition and before the corresponding writer claim is released.
void publish_pipeline_terminal(
    PipelineState &state, PipelineTerminal terminal,
    PipelineClaimAuthority authority = PipelineClaimAuthority::Shared) noexcept;
// Same terminal transition for a caller that already owns both state.gate and
// state.publication->gate. This is the no-relock commit primitive used by the
// product two-bank publication transaction.
void publish_pipeline_terminal_locked(
    PipelineState &state, PipelineTerminal terminal,
    PipelineClaimAuthority authority = PipelineClaimAuthority::Shared,
    bool private_authority_proven = false,
    bool shared_claim_gate_proven = false) noexcept;
// Mutation-free readiness check for a caller that owns state.gate. A true
// result remains stable while that state gate is held because no peer can
// complete this exact attempt through a different PipelineState.
[[nodiscard]] bool
shared_pipeline_terminal_ready(const PipelineState &state) noexcept;
// No-throw single-Pipeline commit used while a higher-level Authority gate is
// held. Publication and Device claim locks stay held until every Pipeline and
// Buffer mutation is complete.
void publish_shared_pipeline_terminal_transaction(
    PipelineState &state, PipelineTerminal terminal) noexcept;
// Product transaction primitive. Callers own both PipelineState gates; this
// function requires nonnull publication/callback preconditions, address-orders
// both publication locks, and exposes neither terminal until the callback has
// committed the associated backing publication.
void publish_private_pipeline_terminal_pair(PipelineState &, PipelineTerminal,
                                            PipelineState &, PipelineTerminal,
                                            void *,
                                            void (*)(void *) noexcept) noexcept;

// Fixed-capacity product transaction for a complete Graph stage set. Callers
// own every PipelineState gate. Publication gates are acquired in address
// order and remain held until every terminal and the backing callback have
// committed, so no observer can see a proper prefix of the stage set.
void publish_private_pipeline_terminal_set(std::span<PipelineState *const>,
                                           std::span<const PipelineTerminal>,
                                           void *,
                                           void (*)(void *) noexcept) noexcept;

// Same pair transaction for ordinary shared Buffer claims. Both publication
// gates and the one Device claim gate stay held through the backing callback.
void publish_shared_pipeline_terminal_pair(PipelineState &, PipelineTerminal,
                                           PipelineState &, PipelineTerminal,
                                           void *,
                                           void (*)(void *) noexcept) noexcept;

class ClaimGuard final {
public:
  ClaimGuard(DeviceState *const device,
             const std::span<const BufferClaim> claims) noexcept
      : device_(device), claims_(claims) {}
  ClaimGuard(DeviceState &device,
             const std::span<const BufferClaim> claims) noexcept
      : ClaimGuard(&device, claims) {}
  ClaimGuard(const ClaimGuard &) = delete;
  ClaimGuard &operator=(const ClaimGuard &) = delete;
  ~ClaimGuard() {
    if (device_ != nullptr) {
      release_claims(*device_, claims_);
    }
  }
  void release() noexcept {
    if (device_ != nullptr) {
      release_claims(*device_, claims_);
      device_ = nullptr;
    }
  }
  // Ownership was discharged by publish_claims under its publication lock.
  void dismiss() noexcept { device_ = nullptr; }

private:
  DeviceState *device_{};
  std::span<const BufferClaim> claims_{};
};

} // namespace rund::compute::detail
