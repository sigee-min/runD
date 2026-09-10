#include "../../registry.hpp"
#include "../graph_forecast.hpp"

#include <algorithm>
#include <exception>

namespace rund::compute::detail::residency {

Authority::~Authority() noexcept {
  for (const auto &holder : cpu_graph_state_.graph_forecast_quarantine) {
    if (holder) {
      std::terminate();
    }
  }
}

} // namespace rund::compute::detail::residency

namespace rund::compute::detail::residency::execution {

GraphForecast &GraphForecast::operator=(GraphForecast &&other) noexcept {
  if (this != &other) {
    if (*this) {
      std::terminate();
    }
    move_from(other);
  }
  return *this;
}

bool GraphForecast::requires_backing() const noexcept {
  return std::any_of(pages_.begin(),
                     pages_.begin() + static_cast<std::ptrdiff_t>(page_count_),
                     [](const GraphForecastPage &page) { return page.fetch; });
}

void GraphForecast::move_from(GraphForecast &other) noexcept {
  owner_ = other.owner_;
  plan_owner_ = std::move(other.plan_owner_);
  pages_ = other.pages_;
  plan_ = other.plan_;
  completion_ = other.completion_;
  token_ = other.token_;
  generation_ = other.generation_;
  coordinate_ = other.coordinate_;
  page_count_ = other.page_count_;
  terminal_ = other.terminal_;
  completion_may_write_ = other.completion_may_write_;
  terminalled_ = other.terminalled_;
  other.clear();
}

void GraphForecast::clear() noexcept {
  owner_ = nullptr;
  plan_owner_.reset();
  pages_ = {};
  plan_ = {};
  completion_ = Status::fail(Reason::CompletionInvalid);
  token_ = 0u;
  generation_ = 0u;
  coordinate_ = 0u;
  page_count_ = 0u;
  terminal_ = TerminalKind::Known;
  completion_may_write_ = false;
  terminalled_ = false;
}

} // namespace rund::compute::detail::residency::execution
