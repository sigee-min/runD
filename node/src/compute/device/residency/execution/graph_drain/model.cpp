#include "../graph_drain.hpp"

#include <utility>

namespace rund::compute::detail::residency::execution {

void GraphDrain::move_from(GraphDrain &other) noexcept {
  owner_ = other.owner_;
  plan_owner_ = std::move(other.plan_owner_);
  pages_ = other.pages_;
  plan_ = other.plan_;
  completion_ = other.completion_;
  source_token_ = other.source_token_;
  destination_token_ = other.destination_token_;
  coordinate_ = other.coordinate_;
  page_count_ = other.page_count_;
  terminal_ = other.terminal_;
  completion_may_write_ = other.completion_may_write_;
  terminalled_ = other.terminalled_;
  other.clear();
}

void GraphDrain::clear() noexcept {
  owner_ = nullptr;
  plan_owner_.reset();
  pages_ = {};
  plan_ = {};
  completion_ = Status::fail(Reason::CompletionInvalid);
  source_token_ = 0u;
  destination_token_ = 0u;
  coordinate_ = 0u;
  page_count_ = 0u;
  terminal_ = TerminalKind::Known;
  completion_may_write_ = false;
  terminalled_ = false;
}

} // namespace rund::compute::detail::residency::execution
