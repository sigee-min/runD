#include "../graph_persist.hpp"

#include <utility>

namespace rund::compute::detail::residency::execution {

void GraphPersist::move_from(GraphPersist &other) noexcept {
  owner_ = other.owner_;
  plan_owner_ = std::move(other.plan_owner_);
  pages_ = other.pages_;
  plan_ = other.plan_;
  identity_ = other.identity_;
  completion_ = other.completion_;
  token_ = other.token_;
  generation_ = other.generation_;
  coordinate_ = other.coordinate_;
  book_domain_ = other.book_domain_;
  region_ = other.region_;
  page_count_ = other.page_count_;
  terminal_ = other.terminal_;
  completion_may_write_ = other.completion_may_write_;
  identity_bad_ = other.identity_bad_;
  terminalled_ = other.terminalled_;
  other.clear();
}

void GraphPersist::clear() noexcept {
  owner_ = nullptr;
  plan_owner_.reset();
  pages_ = {};
  plan_ = {};
  identity_ = {};
  completion_ = Status::fail(Reason::CompletionInvalid);
  token_ = 0u;
  generation_ = 0u;
  coordinate_ = 0u;
  book_domain_ = 0u;
  region_ = {};
  page_count_ = 0u;
  terminal_ = TerminalKind::Known;
  completion_may_write_ = false;
  identity_bad_ = false;
  terminalled_ = false;
}

} // namespace rund::compute::detail::residency::execution
