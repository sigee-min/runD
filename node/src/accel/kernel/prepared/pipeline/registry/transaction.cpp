#include "../registry.hpp"

#include "../reservation.hpp"
#include "internal.hpp"

#include <utility>

namespace rund::node::accel::detail {

PipelineBudgetTransaction::PipelineBudgetTransaction() noexcept = default;

bool PipelineBudgetTransaction::begin(
    PreparedKernelTemplateRegistryState &state,
    PreparedKernelTemplateRegistry &registry) noexcept {
  if (state_ != nullptr) {
    return false;
  }
  lock_ = std::unique_lock<std::recursive_mutex>{state.mutex};
  state_ = &state;
  registry_ = &registry;
  entry_count_ = state.entries.size();
  charge_count_ = state.template_charges.size();
  consumed_ = state.consumed;
  reservation_ = registry.reservation;
  return true;
}

PipelineBudgetTransaction::~PipelineBudgetTransaction() { rollback(); }

void PipelineBudgetTransaction::commit() noexcept {
  committed_ = true;
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
}

void PipelineBudgetTransaction::rollback() noexcept {
  if (state_ == nullptr || committed_) {
    return;
  }
  // All registry publication is serialized by the recursively held lock, so
  // these tails belong exclusively to this failed cold preparation.
  state_->entries.resize(entry_count_);
  state_->template_charges.resize(charge_count_);
  state_->consumed = consumed_;
  registry_->reservation = reservation_;
  committed_ = true;
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
}

} // namespace rund::node::accel::detail
