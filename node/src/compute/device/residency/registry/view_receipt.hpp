#pragma once

#include "../../../pipeline/residency/model.hpp"
#include "cache_model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail::residency {
class Authority;
class ViewOwner;
namespace registration_detail {
class State;
} // namespace registration_detail

namespace registry_model {

// Source-private receipt for one pooled view commit. Row storage is allocated
// during frame registration before any frame-table mutation and reused by
// later commits. The nontrivial lifecycle lives in view_receipt.cpp so this
// record remains a declaration-level model boundary.
class ViewCommitReceipt final {
public:
  ViewCommitReceipt(const ViewCommitReceipt &) = delete;
  ViewCommitReceipt &operator=(const ViewCommitReceipt &) = delete;
  ViewCommitReceipt(ViewCommitReceipt &&) noexcept;
  ViewCommitReceipt &operator=(ViewCommitReceipt &&) = delete;
  ~ViewCommitReceipt();

  [[nodiscard]] Authority *authority() const noexcept { return authority_; }
  [[nodiscard]] bool active() const noexcept { return active_; }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::ViewOwner;

  struct Row final {
    std::uint32_t frame{};
    CacheKey key{};
    DirtyExtent dirty{};
    FrameTier tier{FrameTier::Device};
    FrameRole role{FrameRole::Input};
    FrameState state{FrameState::Empty};
    std::uint64_t extent{};
    std::uint64_t view{};
    std::uint64_t prior_next_use{};
    std::uint64_t prior_retain_until{};
    std::uint64_t post_next_use{};
    std::uint64_t post_retain_until{};
    std::uint64_t stamp{};
    std::uint32_t alias_claims{};
    bool assigned{};
    bool evicted{};
    CacheKey post_key{};
    DirtyExtent post_dirty{};
    FrameState post_state{FrameState::Empty};
    FrameTier post_tier{FrameTier::Device};
    FrameRole post_role{FrameRole::Input};
    std::uint64_t post_extent{};
    std::uint64_t post_view{};
    std::uint32_t post_alias_claims{};
    std::uint64_t transaction_generation{};
    std::uint64_t post_transaction_generation{};
    std::shared_ptr<const registration_detail::State> direct_registration{};
    std::shared_ptr<const registration_detail::State>
        post_direct_registration{};
  };

  ViewCommitReceipt(Authority *, std::uint64_t, std::uint64_t) noexcept;
  void recycle() noexcept;
  void drop() noexcept;

  Authority *authority_{};
  std::uint64_t authority_id_{};
  std::uint64_t stamp_{};
  std::unique_ptr<Row[]> rows_{};
  std::size_t row_capacity_{};
  std::size_t row_count{};
  bool active_{};
  bool quarantined_{};
};

} // namespace registry_model
} // namespace rund::compute::detail::residency
