#pragma once

#include "../local.hpp"

#include "src/compute/device/residency/execution/direct_recurrence/registration.hpp"
#include "src/compute/device/residency/registry.hpp"

#include <rund/compute/reason.hpp>

#include <cstdint>
#include <memory>

namespace rund_node_test_pipeline_residency::service_free_direct_test::
    authority_detail {

namespace residency = rund::compute::detail::residency;
namespace execution = residency::execution;

struct Fixture final {
  std::shared_ptr<residency::Registry> registry{};
  std::shared_ptr<residency::DirectRecurrenceRegistration> registration{};

  [[nodiscard]] bool initialize(std::uint64_t iterations = 5u) noexcept;
  [[nodiscard]] residency::Authority &authority() noexcept;
  [[nodiscard]] residency::DirectRecurrenceRequest request() const noexcept;
  [[nodiscard]] bool release() noexcept;
};

struct Publication final {
  std::uint64_t count{};
  bool success{};
  Fixture *fixture{};
  residency::Authority *authority{};
  residency::DirectRecurrenceLease *lease{};
  bool reenter{};
  bool same_rows{};
  bool rows_before{};
  bool rows_after{};
  bool credential_same{};
  residency::registration_detail::Lifecycle phase_before{
      residency::registration_detail::Lifecycle::Active};
  residency::registration_detail::Lifecycle phase_after{
      residency::registration_detail::Lifecycle::Active};
  residency::DirectAbort abort_result{residency::DirectAbort::Invalid};
};

[[nodiscard]] bool SameSnapshot(
    const residency::DirectRecurrenceRegistration::Snapshot &left,
    const residency::DirectRecurrenceRegistration::Snapshot &right) noexcept;
[[nodiscard]] bool RowsBusy(
    Fixture &fixture,
    const residency::DirectRecurrenceRegistration::Snapshot &snapshot) noexcept;
void Publish(void *raw, bool success) noexcept;
[[nodiscard]] bool Finish(Fixture &fixture,
                          residency::DirectRecurrenceLease &lease,
                          residency::DirectRecurrenceFinal &&final,
                          Publication &publication) noexcept;

[[nodiscard]] bool SuccessCase(std::uint64_t iterations) noexcept;
[[nodiscard]] bool KnownRetryCase() noexcept;
[[nodiscard]] bool KnownPartialRetryCase() noexcept;
[[nodiscard]] bool KnownNoWritePartialRetryCase() noexcept;
[[nodiscard]] bool FrozenBusyCase() noexcept;
[[nodiscard]] bool UnknownAndForgeryCase() noexcept;

} // namespace
  // rund_node_test_pipeline_residency::service_free_direct_test::authority_detail
