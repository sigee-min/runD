#pragma once

#include "../../../../../accel/kernel/prepared/interface/api.hpp"
#include "../../../../device/residency/execution/direct_recurrence/registration.hpp"
#include "../../../../pipeline/state.hpp"
#include "../../../../status.hpp"

#include <atomic>
#include <memory>
#include <optional>

namespace rund::compute::detail::accel_backend::service_free_direct {

struct Run final {
  std::shared_ptr<PipelineState> pipeline{};
  std::shared_ptr<residency::DirectRecurrenceRegistration> registration{};
  std::optional<residency::DirectRecurrenceLease> lease{};
  node::accel::detail::ServiceFreeDirectPreparation preparation{};
  node::accel::detail::ServiceFreeDirectFinal final{};
  std::atomic_bool done{false};
  std::uint64_t callback_count{};
  std::uint64_t publication_count{};
  bool publication_success{};
};

void complete(void *, node::accel::detail::ServiceFreeDirectFinal &&) noexcept;
[[nodiscard]] Status submit(Run &) noexcept;
[[nodiscard]] Status close(Run &) noexcept;
void cancel(Run &, Status) noexcept;

} // namespace rund::compute::detail::accel_backend::service_free_direct
