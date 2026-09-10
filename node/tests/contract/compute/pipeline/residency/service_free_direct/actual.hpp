#pragma once

#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/compute/device/residency/execution/direct_recurrence/registration.hpp"

#include <memory>
#include <utility>

namespace rund_node_test_pipeline_residency::service_free_direct_test {

struct ActualAuthorityRun final {
  std::shared_ptr<rund::compute::detail::residency::Registry> registry{};
  std::shared_ptr<
      rund::compute::detail::residency::DirectRecurrenceRegistration>
      registration{};
  rund::compute::detail::residency::DirectRecurrenceLease lease{};
  std::uint64_t publication_count{};
  bool publication_success{};

  ActualAuthorityRun() = default;
  ActualAuthorityRun(
      std::shared_ptr<rund::compute::detail::residency::Registry>
          registry_value,
      std::shared_ptr<
          rund::compute::detail::residency::DirectRecurrenceRegistration>
          registration_value,
      rund::compute::detail::residency::DirectRecurrenceLease
          &&lease_value) noexcept
      : registry{std::move(registry_value)},
        registration{std::move(registration_value)},
        lease{std::move(lease_value)} {}
  ActualAuthorityRun(const ActualAuthorityRun &) = delete;
  ActualAuthorityRun &operator=(const ActualAuthorityRun &) = delete;
  ActualAuthorityRun(ActualAuthorityRun &&) noexcept = default;
  ActualAuthorityRun &operator=(ActualAuthorityRun &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return registry != nullptr && registration != nullptr && lease;
  }
};

[[nodiscard]] ActualAuthorityRun begin_actual_authority_run(
    std::shared_ptr<rund::compute::detail::residency::Registry> registry,
    std::shared_ptr<const rund::node::accel::detail::ServiceFreeDirectProof>
        proof) noexcept;

[[nodiscard]] bool
finish_actual_authority_run(ActualAuthorityRun &run,
                            rund::compute::Status status) noexcept;

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
