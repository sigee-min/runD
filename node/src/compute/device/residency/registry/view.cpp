#include "../registry.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <new>

namespace rund::compute::detail::residency {

bool Authority::graph_forecast_quarantine_active_locked() const noexcept {
  return std::any_of(cpu_graph_state_.graph_forecast_quarantine.begin(),
                     cpu_graph_state_.graph_forecast_quarantine.end(),
                     [](const execution::GraphForecast &holder) {
                       return static_cast<bool>(holder);
                     });
}

bool Authority::graph_forecast_quarantine_slot_locked() const noexcept {
  return std::any_of(
      cpu_graph_state_.graph_forecast_quarantine.begin(),
      cpu_graph_state_.graph_forecast_quarantine.end(),
      [](const execution::GraphForecast &holder) { return !holder; });
}

registry_model::ViewCommitState
Authority::view_commit_state_locked() const noexcept {
  if (view_state_.view_quarantine != nullptr ||
      graph_forecast_quarantine_active_locked()) {
    return registry_model::ViewCommitState::Quarantined;
  }
  if (view_state_.active_view_commit_stamp != 0u) {
    return registry_model::ViewCommitState::Active;
  }
  return registry_model::ViewCommitState::Idle;
}

bool Authority::view_commit_inflight_locked() const noexcept {
  return view_state_.active_view_commit_stamp != 0u;
}

bool Authority::view_commit_quarantined_locked() const noexcept {
  return view_commit_state_locked() ==
         registry_model::ViewCommitState::Quarantined;
}

bool Authority::view_commit_active_locked(
    const ViewCommitReceipt &receipt) const noexcept {
  return view_commit_state_locked() ==
             registry_model::ViewCommitState::Active &&
         view_state_.active_view_commit_stamp != 0u &&
         view_state_.active_view_commit_stamp == receipt.stamp_ &&
         receipt.active_ && !receipt.quarantined_;
}

bool Authority::ensure_view_commit_storage_locked(
    const std::size_t capacity) noexcept {
  if (capacity == 0u) {
    return false;
  }
  if (view_state_.view_idle != nullptr) {
    if (view_state_.view_idle->authority_ != this ||
        view_state_.view_idle->authority_id_ != credentials_.owner_id ||
        view_state_.view_idle->active_ || view_state_.view_idle->quarantined_ ||
        view_state_.view_idle->stamp_ != 0u ||
        view_state_.view_idle->row_count != 0u) {
      return false;
    }
    if (view_state_.view_idle->rows_ != nullptr &&
        view_state_.view_idle->row_capacity_ >= capacity) {
      return true;
    }
    std::unique_ptr<ViewCommitReceipt::Row[]> rows(
        new (std::nothrow) ViewCommitReceipt::Row[capacity]);
    if (rows == nullptr) {
      return false;
    }
    view_state_.view_idle->rows_ = std::move(rows);
    view_state_.view_idle->row_capacity_ = capacity;
    return true;
  }

  std::unique_ptr<ViewCommitReceipt> idle(
      new (std::nothrow) ViewCommitReceipt(this, credentials_.owner_id, 0u));
  if (idle == nullptr) {
    return false;
  }
  idle->rows_.reset(new (std::nothrow) ViewCommitReceipt::Row[capacity]);
  if (idle->rows_ == nullptr) {
    return false;
  }
  idle->row_capacity_ = capacity;
  view_state_.view_idle = std::move(idle);
  return true;
}

} // namespace rund::compute::detail::residency
