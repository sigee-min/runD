#include "internal.hpp"

namespace rund::compute::detail::residency::execution {

namespace {

[[nodiscard]] bool mint_owner(std::uint64_t &owner) noexcept {
  std::uint64_t current = NextSlidingOwner.load(std::memory_order_relaxed);
  for (;;) {
    if (current == 0u || current == std::numeric_limits<std::uint64_t>::max()) {
      owner = 0u;
      return false;
    }
    if (NextSlidingOwner.compare_exchange_weak(current, current + 1u,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
      owner = current;
      return true;
    }
  }
}

} // namespace

Sliding Sliding::create_internal(SlidingInvocation invocation,
                                 const std::uint64_t authority_token,
                                 const std::uint64_t run_generation,
                                 const std::uint32_t host_input_capacity,
                                 const std::uint32_t host_output_capacity,
                                 const bool model_only) noexcept {
  std::size_t required_inputs = 0u;
  std::size_t required_outputs = 0u;
  std::size_t required_live_inputs = 0u;
  std::size_t physical_inputs = 0u;
  std::size_t physical_outputs = 0u;
  if (!invocation || !invocation.identity() || authority_token == 0u ||
      run_generation == 0u || host_input_capacity == 0u ||
      host_input_capacity > SlidingHostCapacity || host_output_capacity == 0u ||
      host_output_capacity > SlidingHostCapacity ||
      !invocation.capacity_requirements(required_inputs, required_outputs) ||
      !invocation.host_input_requirement(required_live_inputs) ||
      !invocation.host_ring_capacities(physical_inputs, physical_outputs) ||
      host_input_capacity < required_inputs ||
      host_input_capacity < required_live_inputs ||
      host_output_capacity < required_outputs ||
      host_input_capacity > physical_inputs ||
      host_output_capacity > physical_outputs) {
    return {};
  }
  std::uint64_t owner = 0u;
  if (!mint_owner(owner)) {
    return {};
  }
  try {
    return Sliding{std::make_shared<State>(
        std::move(invocation), authority_token, run_generation, owner,
        host_input_capacity, host_output_capacity, required_inputs,
        required_outputs, model_only)};
  } catch (const std::bad_alloc &) {
    return {};
  }
}

Sliding Sliding::create(SlidingInvocation invocation,
                        const std::uint64_t authority_token,
                        const std::uint64_t run_generation,
                        const std::uint32_t host_input_capacity,
                        const std::uint32_t host_output_capacity) noexcept {
  return create_internal(std::move(invocation), authority_token, run_generation,
                         host_input_capacity, host_output_capacity, true);
}

Sliding Sliding::create_bound(
    SlidingInvocation invocation, const std::uint64_t authority_token,
    const std::uint64_t run_generation, const std::uint32_t host_input_capacity,
    const std::uint32_t host_output_capacity) noexcept {
  return create_internal(std::move(invocation), authority_token, run_generation,
                         host_input_capacity, host_output_capacity, false);
}

void Sliding::State::clear_attempt() noexcept {
  inputs = {};
  outputs = {};
  native_cells = {};
  terminals = {};
  status = Status::success();
  first_failure_terminal = TerminalKind::Known;
  terminal = TerminalKind::Known;
  first_failure = {};
  admitted = 0u;
  terminal_frontier = 0u;
  fetch_calls = 0u;
  fetch_hits = 0u;
  promote_calls = 0u;
  drain_calls = 0u;
  persist_calls = 0u;
  persist_issued = 0u;
  persist_frontier = 0u;
  persist_completed_after_failure = 0u;
  fetch_bytes = 0u;
  promote_bytes = 0u;
  drain_bytes = 0u;
  persist_bytes = 0u;
  persist_bytes_after_failure = 0u;
  has_failure = false;
  first_failure_may_write = false;
  quarantine = false;
  authority_bound = false;
  finalizing = false;
  final_nonce = 0u;
  closed = true;
}

bool Sliding::State::rearm_bound(const residency::Identity next_plan,
                                 const std::uint64_t next_token,
                                 const std::uint64_t next_generation) noexcept {
  if (!closed || finalizing || quarantine || next_plan != plan ||
      next_token == 0u || next_generation == 0u || !no_issued() ||
      !callbacks_quiesced() ||
      (has_failure &&
       (first_failure_terminal != TerminalKind::Known ||
        terminal != TerminalKind::Known || first_failure_may_write))) {
    return false;
  }
  clear_attempt();
  token = next_token;
  generation = next_generation;
  return true;
}

bool Sliding::State::abandon_bound() noexcept {
  if (!authority_bound || closed || finalizing || quarantine || has_failure ||
      !no_issued() || !callbacks_quiesced() || admitted != 0u ||
      terminal_frontier != 0u || fetch_calls != 0u || fetch_hits != 0u ||
      promote_calls != 0u || drain_calls != 0u || persist_calls != 0u ||
      persist_issued != 0u || persist_frontier != 0u) {
    return false;
  }
  clear_attempt();
  token = 0u;
  generation = 0u;
  return true;
}

void Sliding::State::quarantine_bound(const Status failure) noexcept {
  mark_unknown(failure);
  finalizing = false;
  final_nonce = 0u;
}

bool Sliding::bind_authority(const residency::Identity plan,
                             const std::uint64_t token,
                             const std::uint64_t generation,
                             std::uint64_t &owner) noexcept {
  owner = 0u;
  if (state_ == nullptr) {
    return false;
  }
  std::lock_guard lock{state_->gate};
  if (state_->model_only || state_->owner == 0u ||
      state_->owner == std::numeric_limits<std::uint64_t>::max() ||
      state_->plan != plan || token == 0u || generation == 0u) {
    return false;
  }
  if (state_->closed && !state_->rearm_bound(plan, token, generation)) {
    return false;
  }
  if (state_->finalizing || state_->authority_bound || state_->token != token ||
      state_->generation != generation || state_->has_failure ||
      state_->quarantine || !state_->no_issued() || state_->admitted != 0u ||
      state_->terminal_frontier != 0u || state_->fetch_calls != 0u ||
      state_->fetch_hits != 0u || state_->promote_calls != 0u ||
      state_->drain_calls != 0u || state_->persist_calls != 0u ||
      state_->persist_issued != 0u || state_->persist_frontier != 0u) {
    return false;
  }
  state_->authority_bound = true;
  state_->closed = false;
  owner = state_->owner;
  return true;
}

bool Sliding::project(const std::uint64_t ordinal,
                      const std::span<residency::PageUse> scratch,
                      SlidingProjection &projection) const noexcept {
  return state_ != nullptr &&
         state_->invocation.project(ordinal, scratch, projection);
}

bool Sliding::quiescent() const noexcept {
  if (state_ == nullptr) {
    return true;
  }
  std::lock_guard lock{state_->gate};
  return !state_->quarantine && !state_->finalizing && state_->no_issued() &&
         (!state_->authority_bound || state_->closed);
}

} // namespace rund::compute::detail::residency::execution
