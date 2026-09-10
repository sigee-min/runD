#pragma once

#include "../../../../../accel/kernel/residency/device_vsm/proof.hpp"
#include "../../pool.hpp"
#include "../../registry/direct_recurrence_owner.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

// Lossless join between one page-coordinate DeviceVsm proof and the existing
// aggregate resident Authority. The Authority owns every typed physical
// backing row; the proof retains the exact backend handles and page geometry.
class DeviceVsmRegistration final
    : public std::enable_shared_from_this<DeviceVsmRegistration> {
public:
  using Proof = node::accel::detail::DeviceVsmProof;
  static constexpr std::size_t Capacity =
      node::accel::detail::DeviceVsmResidentCapacity;

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

  DeviceVsmRegistration(const DeviceVsmRegistration &) = delete;
  DeviceVsmRegistration &operator=(const DeviceVsmRegistration &) = delete;
  ~DeviceVsmRegistration();

  [[nodiscard]] DirectRecurrenceRequest request() const noexcept;
  [[nodiscard]] Snapshot snapshot() const noexcept;
  [[nodiscard]] RegistrationResult
  release_pending(const DirectRecurrenceLease &) noexcept;
  [[nodiscard]] ExecutionClose
  finish_pending(const DirectRecurrenceLease &, void *,
                 DirectRecurrencePublication) noexcept;
  [[nodiscard]] RegistrationResult release() noexcept;

private:
  friend std::shared_ptr<DeviceVsmRegistration>
  register_device_vsm_proof(const std::shared_ptr<Registry> &,
                            std::shared_ptr<const Proof>) noexcept;

  DeviceVsmRegistration() = default;

  [[nodiscard]] RegistrationResult rollback() noexcept;

  std::weak_ptr<Registry> registry_{};
  std::shared_ptr<const Proof> proof_{};
  std::shared_ptr<const registration_detail::State> registration_state_{};
  std::array<ResidentRecurrenceBinding, Capacity> bindings_{};
  std::size_t count_{};
  mutable std::mutex gate_{};
};

[[nodiscard]] std::shared_ptr<DeviceVsmRegistration> register_device_vsm_proof(
    const std::shared_ptr<Registry> &,
    std::shared_ptr<const DeviceVsmRegistration::Proof>) noexcept;

} // namespace rund::compute::detail::residency
