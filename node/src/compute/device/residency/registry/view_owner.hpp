#pragma once

#include "../registry.hpp"

namespace rund::compute::detail::residency {

// Borrows the frame table and receipt storage; never retains a second journal.
class ViewOwner final {
public:
  using Frame = Authority::Frame;
  using ViewCommitReceipt = registry_model::ViewCommitReceipt;

  explicit ViewOwner(Authority &) noexcept;
  ViewOwner(const ViewOwner &) noexcept = default;
  ViewOwner &operator=(const ViewOwner &) = delete;

  [[nodiscard]] ViewActivationResult
      activate_views(std::span<const FrameRegion>) noexcept;
  [[nodiscard]] ViewActivationResult
  commit_view_plan(const registry_model::ViewCommitPlan &,
                   std::unique_ptr<ViewCommitReceipt> *) noexcept;
  [[nodiscard]] bool
  close_view_commit(std::unique_ptr<ViewCommitReceipt> &) noexcept;
  [[nodiscard]] bool
  abort_view_commit(std::unique_ptr<ViewCommitReceipt> &) noexcept;
  [[nodiscard]] bool
  quarantine_view_commit(std::unique_ptr<ViewCommitReceipt> &) noexcept;

private:
  [[nodiscard]] bool
  validate_view_commit(const ViewCommitReceipt &) const noexcept;
  [[nodiscard]] bool
  view_commit_recycle_ready_locked(const ViewCommitReceipt &) const noexcept;
  [[nodiscard]] bool
  recycle_view_commit_locked(std::unique_ptr<ViewCommitReceipt> &) noexcept;
  [[nodiscard]] bool move_view_commit_to_quarantine_locked(
      std::unique_ptr<ViewCommitReceipt> &) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
