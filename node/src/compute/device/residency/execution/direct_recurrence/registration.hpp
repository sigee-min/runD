#pragma once

#include "../../../../../accel/kernel/residency/service_free_direct/proof.hpp"
#include "../../pool.hpp"
#include "../../registry/direct_recurrence_owner.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

// One fixed-size lossless join between a common semantic proof and the sole
// Device Authority. The proof owns every backend resident handle; this owner
// owns only Authority registrations and releases them when the prepared
// Pipeline is retired. Unknown execution retains this owner through the
// Authority slot, deliberately keeping the registrations quarantined.
class DirectRecurrenceRegistration final
    : public std::enable_shared_from_this<DirectRecurrenceRegistration> {
public:
  using Proof = node::accel::detail::ServiceFreeDirectProof;
  static constexpr std::size_t Capacity = Proof::StateCapacity;

  struct Snapshot final {
    std::shared_ptr<const Proof> proof{};
    std::shared_ptr<const registration_detail::State> state{};
    std::array<ResidentRecurrenceBinding, Capacity> bindings{};
    std::size_t count{};

    [[nodiscard]] std::span<const ResidentRecurrenceBinding>
    binding_span() const noexcept {
      return {bindings.data(), count};
    }
  };

  DirectRecurrenceRegistration(const DirectRecurrenceRegistration &) = delete;
  DirectRecurrenceRegistration &
  operator=(const DirectRecurrenceRegistration &) = delete;
  ~DirectRecurrenceRegistration();

  [[nodiscard]] DirectRecurrenceRequest request() const noexcept;
  [[nodiscard]] Snapshot snapshot() const noexcept;
  [[nodiscard]] RegistrationResult
  release_pending(const DirectRecurrenceLease &) noexcept;
  [[nodiscard]] ExecutionClose
  finish_pending(const DirectRecurrenceLease &, void *,
                 DirectRecurrencePublication) noexcept;
  [[nodiscard]] RegistrationResult release() noexcept;

private:
  friend std::shared_ptr<DirectRecurrenceRegistration>
  register_direct_recurrence_proof(
      const std::shared_ptr<Registry> &,
      std::shared_ptr<
          const node::accel::detail::ServiceFreeDirectProof>) noexcept;

  DirectRecurrenceRegistration() = default;

  [[nodiscard]] RegistrationResult rollback() noexcept;

  std::weak_ptr<Registry> registry_{};
  std::shared_ptr<const Proof> proof_{};
  std::shared_ptr<const registration_detail::State> registration_state_{};
  std::array<ResidentRecurrenceBinding, Capacity> bindings_{};
  std::size_t count_{};
  mutable std::mutex gate_{};
};

[[nodiscard]] std::shared_ptr<DirectRecurrenceRegistration>
register_direct_recurrence_proof(
    const std::shared_ptr<Registry> &registry,
    std::shared_ptr<const node::accel::detail::ServiceFreeDirectProof>
        proof) noexcept;

} // namespace rund::compute::detail::residency
