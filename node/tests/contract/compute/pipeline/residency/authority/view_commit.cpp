#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"
#include "src/compute/device/residency/registry/view_owner.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityViewCommit() {
  using namespace rund::compute::detail::residency;

  Authority authority;
  std::uint32_t first = 99u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                 first) ||
      first != 0u) {
    return 1;
  }
  const FrameRegion region{.tier = FrameTier::Device,
                           .role = FrameRole::Input,
                           .first = first,
                           .count = 2u};
  const auto never = std::numeric_limits<std::uint64_t>::max();
  const std::array uses{
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 0u},
               .access = Access::Read,
               .next_use = never},
      CacheUse{.key = {.backing = 7u, .version = 1u, .page = 1u},
               .access = Access::Read,
               .next_use = never},
  };
  const AuthorityResult seed = authority.begin(uses, region.tier, region.role,
                                               region.first, region.count);
  if (!seed || !authority.complete(seed.lease.token, true)) {
    return 2;
  }

  const StreamPlan stream{2u, 2u, DirtyRange{.bytes = 4096u}, 8192u, 1u};
  const std::array regions{region};
  const registry_model::ViewCommitPlan plan{
      .stream_regions = regions,
      .stream = &stream,
      .last = uses[1].key,
  };
  auto views = authority.views();
  std::unique_ptr<registry_model::ViewCommitReceipt> receipt;
  if (!views.commit_view_plan(plan, &receipt) || receipt == nullptr ||
      !receipt->active() || receipt->authority() != &authority) {
    if (receipt != nullptr) {
      static_cast<void>(views.quarantine_view_commit(receipt));
    }
    return 3;
  }
  auto *const holder = receipt.get();

  // A live receipt owns the view transition. Failed cleanup attempts must not
  // consume it or make its destructor observe a detached active record.
  std::unique_ptr<registry_model::ViewCommitReceipt> second;
  std::uint32_t rejected = 99u;
  if (views.commit_view_plan(plan, &second).failure != AuthorityFailure::Busy ||
      second != nullptr ||
      views.commit_view_plan(plan, &receipt).failure !=
          AuthorityFailure::Busy ||
      receipt.get() != holder || !receipt->active() ||
      authority.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                rejected)) {
    static_cast<void>(views.quarantine_view_commit(receipt));
    return 4;
  }

  // Close and Known abort both return the same preallocated receipt holder to
  // the Authority idle slot; neither path allocates a replacement.
  if (!views.close_view_commit(receipt) || receipt != nullptr ||
      !views.commit_view_plan(plan, &receipt) || receipt.get() != holder ||
      !views.abort_view_commit(receipt) || receipt != nullptr ||
      !views.commit_view_plan(plan, &receipt) || receipt.get() != holder) {
    if (receipt != nullptr) {
      static_cast<void>(views.quarantine_view_commit(receipt));
    }
    return 5;
  }

  Authority foreign;
  auto foreign_views = foreign.views();
  const auto remains_authentic = [&]() noexcept {
    return receipt != nullptr && receipt.get() == holder && receipt->active() &&
           receipt->authority() == &authority;
  };
  if (!remains_authentic() || foreign_views.close_view_commit(receipt) ||
      !remains_authentic() || foreign_views.abort_view_commit(receipt) ||
      !remains_authentic() || foreign_views.quarantine_view_commit(receipt) ||
      !remains_authentic() ||
      foreign_views.commit_view_plan(plan, &receipt).failure !=
          AuthorityFailure::Invalid ||
      !remains_authentic()) {
    static_cast<void>(views.quarantine_view_commit(receipt));
    return 6;
  }

  // Unknown disposition is sticky: the receipt is owned by Authority's
  // quarantine holder, and setup or another view commit cannot proceed.
  if (!views.quarantine_view_commit(receipt) || receipt != nullptr) {
    if (receipt != nullptr) {
      static_cast<void>(views.quarantine_view_commit(receipt));
    }
    return 7;
  }
  std::uint32_t post_quarantine = 99u;
  std::unique_ptr<registry_model::ViewCommitReceipt> post_receipt;
  if (authority.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                post_quarantine) ||
      views.commit_view_plan(plan, &post_receipt).failure !=
          AuthorityFailure::Busy ||
      post_receipt != nullptr) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
