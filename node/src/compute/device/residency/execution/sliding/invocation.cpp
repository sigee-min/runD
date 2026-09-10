#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

SlidingInvocation
SlidingInvocation::direct(std::shared_ptr<const Plan> plan) noexcept {
  SlidingInvocation result{};
  if (plan != nullptr && plan->identity() != 0u && plan->epoch_count() != 0u) {
    result.topology_ = SlidingTopology::Direct;
    result.direct_ = std::move(plan);
  }
  return result;
}

SlidingInvocation SlidingInvocation::graph(
    std::shared_ptr<const residency::ResidencyPlan> plan,
    const std::uint64_t page_count,
    const std::span<const std::uint64_t> logical_bytes) noexcept {
  SlidingInvocation result{};
  if (plan != nullptr && plan->graph_tiled() &&
      plan->tiled_graph().active(page_count, logical_bytes, result.graph_)) {
    result.topology_ = SlidingTopology::Graph;
    result.graph_owner_ = std::move(plan);
  }
  return result;
}

SlidingInvocation::operator bool() const noexcept {
  return topology_ == SlidingTopology::Direct
             ? direct_ != nullptr && direct_->identity() != 0u
             : topology_ == SlidingTopology::Graph && graph_owner_ != nullptr &&
                   graph_.valid() && graph_owner_->identity();
}

residency::Identity SlidingInvocation::identity() const noexcept {
  if (topology_ == SlidingTopology::Direct && direct_ != nullptr) {
    return residency::Identity{.lo = direct_->identity()};
  }
  return topology_ == SlidingTopology::Graph && graph_owner_ != nullptr
             ? graph_owner_->identity()
             : residency::Identity{};
}

std::uint64_t SlidingInvocation::count() const noexcept {
  return topology_ == SlidingTopology::Direct
             ? (direct_ == nullptr ? 0u : direct_->epoch_count())
         : topology_ == SlidingTopology::Graph ? graph_.epoch_count()
                                               : 0u;
}

} // namespace rund::compute::detail::residency::execution
