#include "internal.hpp"

namespace rund::compute::detail::residency {

bool GraphForecastOwner::validate_graph_forecast_locked(
    const execution::GraphForecast &ticket) const noexcept {
  if (!ticket || ticket.owner_ != &authority_ ||
      ticket.plan_owner_ == nullptr || ticket.plan_ == Identity{} ||
      ticket.plan_owner_->identity() != ticket.plan_ || ticket.token_ == 0u ||
      ticket.generation_ == 0u || ticket.page_count_ == 0u ||
      ticket.page_count_ > execution::GraphForecastCapacity) {
    return false;
  }
  const auto epoch = std::find_if(
      authority_.cycle_state_.epochs.begin(),
      authority_.cycle_state_.epochs.end(),
      [token = ticket.token_](const registry_model::LeaseSlot &slot) {
        return slot.token == token;
      });
  if (epoch == authority_.cycle_state_.epochs.end() ||
      epoch->state != registry_model::LeaseState::Computing ||
      epoch->cycle != 0u || epoch->cpu_key ||
      epoch->generation != ticket.generation_ ||
      epoch->coordinate != ticket.coordinate_ ||
      epoch->bindings.size() != ticket.page_count_ ||
      std::any_of(epoch->transitions.begin(), epoch->transitions.end(),
                  [](const CacheTransition transition) {
                    return transition.kind == TransitionKind::Writeback;
                  })) {
    return false;
  }
  for (std::size_t index = 0u; index < ticket.page_count_; ++index) {
    const execution::GraphForecastPage &page = ticket.pages_[index];
    const CacheBinding &binding = epoch->bindings[index];
    if (binding.access != Access::Read || binding.frame != page.frame ||
        binding.key != page.key || binding.fetch != page.fetch ||
        page.frame >= authority_.frames_.size() ||
        authority_.frames_[page.frame].state != FrameState::Pinned ||
        authority_.frames_[page.frame].key != page.key) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (ticket.pages_[prior].frame == page.frame ||
          ticket.pages_[prior].key == page.key) {
        return false;
      }
    }
  }
  return true;
}

bool GraphForecastOwner::terminal_graph_forecast(
    execution::GraphForecast &ticket, const Status status,
    const execution::TerminalKind terminal, const bool may_write,
    const std::span<const execution::GraphForecastCompletion>
        completions) noexcept {
  if (ticket.terminalled_ || completions.size() != ticket.page_count_) {
    return false;
  }
  std::lock_guard lock{authority_.gate_};
  if (!validate_graph_forecast_locked(ticket)) {
    return false;
  }
  const AuthorityResult live = authority_.resume_locked(ticket.token_);
  if (!live || live.lease.bindings.size() != ticket.page_count_ ||
      std::any_of(live.lease.transitions.begin(), live.lease.transitions.end(),
                  [](const CacheTransition transition) {
                    return transition.kind == TransitionKind::Writeback;
                  })) {
    return false;
  }
  bool exact = true;
  for (std::size_t index = 0u; index < ticket.page_count_; ++index) {
    const execution::GraphForecastPage page = ticket.pages_[index];
    const execution::GraphForecastCompletion completion = completions[index];
    const CacheBinding binding = live.lease.bindings[index];
    if (binding.frame >= authority_.frames_.size() ||
        authority_.frames_[binding.frame].state != FrameState::Pinned ||
        authority_.frames_[binding.frame].key != page.key) {
      return false;
    }
    exact = exact && completion.key == page.key &&
            completion.frame == page.frame && binding.key == page.key &&
            binding.frame == page.frame && binding.fetch == page.fetch &&
            (page.fetch ? (status ? completion.bytes == page.bytes
                                  : completion.bytes <= page.bytes)
                        : completion.bytes == 0u);
  }
  const bool contradictory =
      terminal == execution::TerminalKind::UnknownMayWrite
          ? static_cast<bool>(status)
          : false;
  ticket.completion_ = exact && !contradictory
                           ? status
                           : Status::fail(Reason::CompletionInvalid);
  ticket.terminal_ = terminal;
  ticket.completion_may_write_ =
      may_write || !exact || contradictory ||
      terminal == execution::TerminalKind::UnknownMayWrite;
  ticket.terminalled_ = true;
  return true;
}

bool GraphForecastOwner::abort_graph_forecast(
    execution::GraphForecast &ticket) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (!validate_graph_forecast_locked(ticket)) {
    return false;
  }
  if (!authority_.complete_locked(ticket.token_, false, true)) {
    return false;
  }
  ticket.clear();
  return true;
}

} // namespace rund::compute::detail::residency
