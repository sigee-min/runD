#include "../freeze/internal.hpp"
#include "../view_owner.hpp"

#include "../../pool.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

namespace view_plan_detail {

bool direct_in_regions(const std::vector<Authority::Frame> &frames,
                       const std::span<const FrameRegion> regions) noexcept {
  for (const FrameRegion region : regions) {
    if (region.first > frames.size() ||
        region.count > frames.size() - region.first) {
      continue;
    }
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      if (frames[index].direct_registration != nullptr) {
        return true;
      }
    }
  }
  return false;
}

bool evicts_frame(const std::vector<Authority::Frame> &frames,
                  const registry_model::ViewCommitPlan &plan,
                  const std::size_t frame) noexcept {
  if (plan.kind != registry_model::ViewCommitPlan::Kind::Graph) {
    return false;
  }
  for (const FrameRegion region : plan.activation_regions) {
    if (region.first >= frames.size()) {
      continue;
    }
    const Authority::Frame &selected = frames[region.first];
    const Authority::Frame &candidate = frames[frame];
    if (selected.extent != 0u && candidate.direct_registration == nullptr &&
        candidate.assigned && candidate.extent == selected.extent &&
        candidate.view != selected.view &&
        candidate.state != FrameState::Empty && candidate.alias_claims == 0u) {
      return true;
    }
  }
  return false;
}

bool touches_frame(const std::vector<Authority::Frame> &frames,
                   const registry_model::ViewCommitPlan &plan,
                   const std::size_t frame) noexcept {
  if (plan.kind == registry_model::ViewCommitPlan::Kind::Stream) {
    return frames[frame].direct_registration == nullptr &&
           in_regions(plan.stream_regions, frame);
  }
  if (frames[frame].direct_registration != nullptr) {
    return false;
  }
  if (in_regions(plan.activation_regions, frame) ||
      evicts_frame(frames, plan, frame)) {
    return true;
  }
  return std::any_of(plan.requests.begin(), plan.requests.end(),
                     [frame](const GraphFreezeRequest &request) {
                       return in_regions(request.regions, frame);
                     });
}

} // namespace view_plan_detail

ViewActivationResult ViewOwner::commit_view_plan(
    const registry_model::ViewCommitPlan &plan,
    std::unique_ptr<ViewCommitReceipt> *const receipt) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (receipt == nullptr) {
    return ViewActivationResult{.failure = AuthorityFailure::Invalid};
  }
  const bool epoch_active =
      std::any_of(authority_.cycle_state_.epochs.begin(),
                  authority_.cycle_state_.epochs.end(), active);
  if (authority_.view_commit_state_locked() !=
          registry_model::ViewCommitState::Idle ||
      authority_.execution_state_.slot.token != 0u ||
      active(authority_.cycle_state_.writeback) ||
      any_active(authority_.cycle_state_.graph_persists) || epoch_active) {
    return ViewActivationResult{.failure = AuthorityFailure::Busy};
  }
  if (*receipt != nullptr) {
    return ViewActivationResult{.failure = AuthorityFailure::Invalid};
  }
  std::uint64_t evictions = 0u;
  if (plan.kind == registry_model::ViewCommitPlan::Kind::Stream) {
    if (view_plan_detail::direct_in_regions(authority_.frames_,
                                            plan.stream_regions)) {
      return ViewActivationResult{.failure = AuthorityFailure::Busy};
    }
    if (plan.stream == nullptr || plan.graph != nullptr ||
        !plan.activation_regions.empty() || plan.stream_regions.empty() ||
        plan.stream_regions.size() > Pool::BankCount + 1u ||
        !plan.requests.empty() ||
        !view_plan_detail::validate_stream(authority_.frames_, *plan.stream,
                                           plan.last, plan.stream_regions)) {
      return ViewActivationResult{.failure = AuthorityFailure::Invalid};
    }
  } else if (plan.kind == registry_model::ViewCommitPlan::Kind::Graph) {
    if (view_plan_detail::direct_in_regions(authority_.frames_,
                                            plan.activation_regions)) {
      return ViewActivationResult{.failure = AuthorityFailure::Busy};
    }
    for (const GraphFreezeRequest &request : plan.requests) {
      if (view_plan_detail::direct_in_regions(authority_.frames_,
                                              request.regions)) {
        return ViewActivationResult{.failure = AuthorityFailure::Busy};
      }
    }
    if (plan.stream != nullptr || plan.graph == nullptr ||
        !plan.stream_regions.empty() ||
        plan.activation_regions.size() >
            TiledGraphResourceCapacity * Pool::BankCount ||
        plan.requests.size() >
            registry_model::ViewCommitPlan::GraphRequestCapacity) {
      return ViewActivationResult{.failure = AuthorityFailure::Invalid};
    }
    for (const GraphFreezeRequest &request : plan.requests) {
      if (request.regions.size() > GraphFreezeRequest::RegionCapacity) {
        return ViewActivationResult{.failure = AuthorityFailure::Invalid};
      }
    }
    const ViewActivationResult checked = view_plan_detail::validate_regions(
        authority_.frames_, plan.activation_regions);
    if (!checked) {
      return checked;
    }
    const bool empty = plan.graph->page_count() == 0u && plan.requests.empty();
    if (!empty && !view_plan_detail::validate_graph(
                      authority_.frames_, *plan.graph, plan.requests)) {
      return ViewActivationResult{.failure = AuthorityFailure::Invalid};
    }
  } else {
    return ViewActivationResult{.failure = AuthorityFailure::Invalid};
  }

  std::size_t touched = 0u;
  for (std::size_t index = 0u; index < authority_.frames_.size(); ++index) {
    if (view_plan_detail::touches_frame(authority_.frames_, plan, index)) {
      ++touched;
    }
  }
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (touched == 0u || touched > authority_.frames_.size() ||
      authority_.view_state_.next_view_stamp == 0u ||
      authority_.view_state_.next_view_stamp == max) {
    return ViewActivationResult{.failure = AuthorityFailure::Capacity};
  }
  if (authority_.view_state_.view_idle == nullptr ||
      authority_.view_state_.view_idle->authority_ != &authority_ ||
      authority_.view_state_.view_idle->authority_id_ !=
          authority_.credentials_.owner_id ||
      authority_.view_state_.view_idle->active_ ||
      authority_.view_state_.view_idle->quarantined_ ||
      authority_.view_state_.view_idle->stamp_ != 0u ||
      authority_.view_state_.view_idle->rows_ == nullptr ||
      authority_.view_state_.view_idle->row_count != 0u ||
      authority_.view_state_.view_idle->row_capacity_ < touched) {
    return ViewActivationResult{.failure = AuthorityFailure::Capacity};
  }
  std::unique_ptr<ViewCommitReceipt> pending =
      std::move(authority_.view_state_.view_idle);
  const std::uint64_t stamp = authority_.view_state_.next_view_stamp;
  pending->stamp_ = stamp;
  pending->active_ = true;
  pending->quarantined_ = false;
  std::size_t row_count = 0u;
  for (std::size_t index = 0u; index < authority_.frames_.size(); ++index) {
    if (!view_plan_detail::touches_frame(authority_.frames_, plan, index)) {
      continue;
    }
    const Frame &frame = authority_.frames_[index];
    ViewCommitReceipt::Row &row = pending->rows_[row_count];
    row.frame = static_cast<std::uint32_t>(index);
    row.key = frame.key;
    row.dirty = frame.dirty;
    row.tier = frame.tier;
    row.role = frame.role;
    row.state = frame.state;
    row.extent = frame.extent;
    row.view = frame.view;
    row.prior_next_use = frame.next_use;
    row.prior_retain_until = frame.retain_until;
    row.stamp = stamp;
    row.alias_claims = frame.alias_claims;
    row.assigned = frame.assigned;
    row.evicted =
        view_plan_detail::evicts_frame(authority_.frames_, plan, index);
    row.transaction_generation = frame.transaction_generation;
    row.direct_registration = frame.direct_registration;
    ++row_count;
  }
  if (row_count != touched) {
    pending->recycle();
    authority_.view_state_.view_idle = std::move(pending);
    return ViewActivationResult{.failure = AuthorityFailure::Invalid};
  }
  pending->row_count = row_count;
  authority_.view_state_.next_view_stamp =
      authority_.view_state_.next_view_stamp + 1u;

  if (plan.kind == registry_model::ViewCommitPlan::Kind::Stream) {
    view_plan_detail::apply_stream(authority_.frames_, *plan.stream, plan.last,
                                   plan.stream_regions);
  } else {
    evictions = view_plan_detail::apply_regions(authority_.frames_,
                                                plan.activation_regions);
    if (!plan.requests.empty()) {
      view_plan_detail::apply_graph(authority_.frames_, *plan.graph,
                                    plan.requests);
    }
  }
  for (std::size_t index = 0u; index < pending->row_count; ++index) {
    ViewCommitReceipt::Row &row = pending->rows_[index];
    Frame &frame = authority_.frames_[row.frame];
    row.post_key = frame.key;
    row.post_dirty = frame.dirty;
    row.post_state = frame.state;
    row.post_tier = frame.tier;
    row.post_role = frame.role;
    row.post_extent = frame.extent;
    row.post_view = frame.view;
    row.post_next_use = frame.next_use;
    row.post_retain_until = frame.retain_until;
    row.post_alias_claims = frame.alias_claims;
    row.post_transaction_generation = frame.transaction_generation;
    row.post_direct_registration = frame.direct_registration;
    frame.view_commit_stamp = row.stamp;
  }
  authority_.view_state_.active_view_commit_stamp = stamp;
  *receipt = std::move(pending);
  return ViewActivationResult{.failure = AuthorityFailure::None,
                              .eviction_count = evictions};
}

} // namespace rund::compute::detail::residency
