#include "../graph_ready.hpp"

#include <utility>

namespace rund::compute::detail::residency::execution {

void GraphReady::move_from(GraphReady &other) noexcept {
  owner_ = other.owner_;
  plan_owner_ = std::move(other.plan_owner_);
  pages_ = other.pages_;
  plan_ = other.plan_;
  coordinate_ = other.coordinate_;
  page_count_ = other.page_count_;
  other.clear();
}

void GraphReady::clear() noexcept {
  owner_ = nullptr;
  plan_owner_.reset();
  pages_ = {};
  plan_ = {};
  coordinate_ = 0u;
  page_count_ = 0u;
}

} // namespace rund::compute::detail::residency::execution
