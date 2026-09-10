#pragma once

#include "../local.hpp"
#include "../sliding/internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../runtime/counter.hpp"
#include "../../../../timeline/owner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <span>

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint64_t TimeoutNs = 30'000'000'000u;
inline constexpr std::uint64_t OwnerMagic = 0x56'50'53'4c'49'44'45'31ull;
inline constexpr std::size_t BatchCount = 2u;

struct Owner final {
  std::uint64_t magic{OwnerMagic};
  std::uint64_t owner_nonce{};
  PersistentResidencySlidingRequest prepared{};
  PersistentResidencySlidingRequest pending_request{};
  bool native_pending{};
  bool quarantine_ticket{};
  PersistentResidencySlidingCapability capability{};
  VulkanPipeline *first{};
  VulkanResidencyPersistentRun *run{};
  // Chunked submissions retain only the two reusable bank batches. A
  // OneSubmit request uses a short-lived submit list at the call boundary.
  std::array<VulkanTimelineBatch, BatchCount> batches{};
};

[[nodiscard]] inline rund::AccelCheck invalid() noexcept {
  return {false, "accel_kernel_pipeline_invalid"};
}

[[nodiscard]] inline rund::AccelCheck unavailable() noexcept {
  return {false, "accel_vulkan_command_unavailable"};
}

// Callers hold Control->Run->Adapter in this order.  This helper folds the
// one physical Unknown fact into every canonical owner without taking a lock
// or clearing an accepted prefix.
void quarantine_unknown_locked(PersistentResidencySlidingControl &, Owner *,
                               VulkanResidencyPersistentRun *, VulkanAdapter *,
                               rund::AccelCheck) noexcept;

[[nodiscard]] std::shared_ptr<Owner>
owner_of(const std::shared_ptr<void> &) noexcept;
[[nodiscard]] bool
same_prepared_request(const Owner &,
                      const PersistentResidencySlidingRequest &) noexcept;
[[nodiscard]] PersistentResidencySlidingCapability
capability_for(std::span<const PersistentResidencySlidingRole>, std::uint64_t,
               ResidencySlidingMemory, PersistentResidencySlidingMode) noexcept;
[[nodiscard]] PersistentResidencySlidingSubmitResult
submit_result(const PersistentResidencySlidingRequest &,
              PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] PersistentResidencySlidingSubmitResult
    submit_out(rund::AccelCheck,
               PersistentResidencySlidingSubmitEvent =
                   PersistentResidencySlidingSubmitEvent::Declined) noexcept;
void reset_control_unlocked(PersistentResidencySlidingControl &) noexcept;
void destroy_attempt_roles(const PersistentResidencySlidingRequest &) noexcept;
void clear_attempt_claims(const PersistentResidencySlidingRequest &,
                          VulkanResidencyPersistentRun *) noexcept;
[[nodiscard]] PersistentResidencySlidingSubmitResult
submit_continuation(const PersistentResidencySlidingRequest &,
                    PersistentResidencySlidingControl &, Owner &,
                    VulkanAdapter &, VulkanResidencyPersistentRun &) noexcept;
[[nodiscard]] PersistentResidencySlidingSubmitResult
submit_initial(const PersistentResidencySlidingRequest &,
               PersistentResidencySlidingControl &,
               const PersistentResidencySlidingCapability &, Owner &,
               VulkanAdapter &, VulkanResidencyPersistentRun &) noexcept;

[[nodiscard]] inline bool
ticket_ready(const Owner &owner,
             const PersistentResidencySlidingRequest &request) noexcept {
  const auto &ticket = request.ticket;
  if (ticket == nullptr || ticket->cell == nullptr ||
      ticket->active != ticket->pending.has_value()) {
    return false;
  }
  if (ticket->active) {
    return ticket->pending.has_value() &&
           persistent_sliding_request_equal(*ticket->pending, request);
  }
  return owner.prepared.mode == request.mode &&
         same_prepared_request(owner, request);
}

[[nodiscard]] inline bool
stage_rearm(Owner &owner,
            const PersistentResidencySlidingRequest &request) noexcept {
  if (owner.native_pending || owner.quarantine_ticket ||
      request.owner_nonce == 0u || request.ticket == nullptr ||
      !ticket_ready(owner, request) ||
      !persistent_sliding_stage_ticket(request.ticket, owner.capability,
                                       request)) {
    return false;
  }
  owner.pending_request = request;
  owner.pending_request.lowering.reset();
  owner.native_pending = true;
  return true;
}

[[nodiscard]] inline bool commit_rearm(Owner &owner) noexcept {
  if (!owner.native_pending) {
    return false;
  }
  const PersistentResidencySlidingRequest next = owner.pending_request;
  if (next.ticket == nullptr ||
      !persistent_sliding_commit_ticket(next.ticket, &owner.prepared)) {
    owner.quarantine_ticket = true;
    persistent_sliding_quarantine_ticket(next.ticket);
    return false;
  }
  owner.prepared = next;
  owner.prepared.lowering.reset();
  owner.owner_nonce = next.owner_nonce;
  owner.first = static_cast<VulkanPipeline *>(
      owner.pending_request.roles[0u].prepared.get());
  owner.run = owner.first == nullptr || owner.first->residency == nullptr
                  ? nullptr
                  : &owner.first->residency->persistent;
  owner.pending_request = {};
  owner.native_pending = false;
  return owner.first != nullptr && owner.run != nullptr;
}

inline void abort_rearm(Owner &owner) noexcept {
  if (owner.native_pending && owner.pending_request.ticket != nullptr) {
    persistent_sliding_abort_ticket(owner.pending_request.ticket);
  }
  owner.pending_request = {};
  owner.native_pending = false;
}

[[nodiscard]] bool control_generation(const PersistentResidencySlidingRole &,
                                      std::uint64_t, std::uint32_t &) noexcept;
[[nodiscard]] bool descriptor_generation(const PersistentResidencySlidingRole &,
                                         std::uint64_t,
                                         std::uint64_t &) noexcept;
[[nodiscard]] bool materialize(
    const PersistentResidencySlidingRequest &,
    std::array<VulkanResidencyPersistentRole, ResidencySlidingCapacity> &,
    VulkanResidencyPersistentRole &, VulkanAdapter *&,
    VulkanResidencyPersistentRun *&,
    VulkanResidencyPersistentRun *expected_run) noexcept;
[[nodiscard]] VulkanTimelinePoint point(const VulkanResidencyPersistentRun &,
                                        std::uint64_t) noexcept;
[[nodiscard]] const PersistentResidencySlidingRole &
source_role(const VulkanResidencyPersistentRun &, std::uint64_t) noexcept;
[[nodiscard]] VulkanResidencyPersistentRole &
native_role(VulkanResidencyPersistentRun &, std::uint64_t) noexcept;
[[nodiscard]] std::size_t local_count(const VulkanResidencyPersistentRun &,
                                      std::uint64_t) noexcept;
void clear_active(VulkanResidencyPersistentRun &) noexcept;
void publish_final(VulkanResidencyPersistentRun &) noexcept;
void close_known(VulkanResidencyPersistentRun &) noexcept;
[[nodiscard]] bool gate_result(VulkanResidencySelection &, std::uint64_t,
                               bool &, const char *&) noexcept;
[[nodiscard]] VulkanResidencyPersistentRun *
active_run(PersistentResidencySlidingControl &) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
