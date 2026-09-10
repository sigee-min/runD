#include "cycle_owner.hpp"

#include "../registry.hpp"
#include "internal.hpp"
#include "lease_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] bool cycle_range(const std::vector<registry_model::Frame> &frames,
                               const cycle::Epoch::Range range) noexcept {
  if (range.count == 0u) {
    return range.mask == 0u;
  }
  if (range.count > 32u || range.mask == 0u ||
      (range.count < 32u && (range.mask >> range.count) != 0u) ||
      range.first > frames.size() ||
      range.count > frames.size() - range.first) {
    return false;
  }
  const FrameTier tier =
      range.domain == cycle::Domain::Host ? FrameTier::Host : FrameTier::Device;
  for (std::size_t index = range.first;
       index < static_cast<std::size_t>(range.first) + range.count; ++index) {
    const std::uint32_t local = static_cast<std::uint32_t>(index - range.first);
    if ((range.mask & (std::uint32_t{1u} << local)) != 0u &&
        (!frames[index].assigned || frames[index].tier != tier)) {
      return false;
    }
  }
  return true;
}

} // namespace

CycleOwner::CycleOwner(Authority &authority) noexcept : authority_(authority) {}

bool CycleOwner::complete_cycle(const std::uint64_t cycle_token,
                                const std::uint64_t epoch_token,
                                const bool success,
                                const bool invalidate_all) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || cycle_token == 0u ||
      retry_ready(authority_.cycle_state_.graph_persists) ||
      authority_.cycle_state_.cycle.token != cycle_token || epoch_token == 0u) {
    return false;
  }
  std::size_t member = authority_.cycle_state_.cycle.count;
  for (std::size_t index = 0u; index < authority_.cycle_state_.cycle.count;
       ++index) {
    if (authority_.cycle_state_.cycle.tokens[index] == epoch_token) {
      member = index;
      break;
    }
  }
  const auto epoch = std::find_if(
      authority_.cycle_state_.epochs.begin(),
      authority_.cycle_state_.epochs.end(),
      [cycle_token, epoch_token](const registry_model::LeaseSlot &slot) {
        return slot.token == epoch_token && slot.cycle == cycle_token;
      });
  if (member == authority_.cycle_state_.cycle.count ||
      epoch == authority_.cycle_state_.epochs.end() || epoch->cpu_key ||
      !complete_epoch(authority_.frames_, *epoch, success, invalidate_all)) {
    return false;
  }
  authority_.cycle_state_.cycle
      .terminals[authority_.cycle_state_.cycle.banks[member]] =
      authority_.cycle_state_.cycle.ordinals[member];
  for (std::size_t index = member + 1u;
       index < authority_.cycle_state_.cycle.count; ++index) {
    authority_.cycle_state_.cycle.tokens[index - 1u] =
        authority_.cycle_state_.cycle.tokens[index];
    authority_.cycle_state_.cycle.ordinals[index - 1u] =
        authority_.cycle_state_.cycle.ordinals[index];
    authority_.cycle_state_.cycle.banks[index - 1u] =
        authority_.cycle_state_.cycle.banks[index];
  }
  --authority_.cycle_state_.cycle.count;
  authority_.cycle_state_.cycle.tokens[authority_.cycle_state_.cycle.count] =
      0u;
  if (!success) {
    authority_.cycle_state_.cycle.closed = true;
  }
  if (authority_.cycle_state_.cycle.closed &&
      authority_.cycle_state_.cycle.count == 0u) {
    authority_.cycle_state_.retired_cycle = authority_.cycle_state_.cycle.token;
    authority_.cycle_state_.cycle = registry_model::CycleSlot{};
  }
  return true;
}

bool CycleOwner::close_cycle(const std::uint64_t token) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || token == 0u ||
      retry_ready(authority_.cycle_state_.graph_persists)) {
    return false;
  }
  if (authority_.cycle_state_.cycle.token == 0u) {
    return authority_.cycle_state_.retired_cycle == token;
  }
  if (authority_.cycle_state_.cycle.token != token) {
    return false;
  }
  authority_.cycle_state_.cycle.closed = true;
  if (authority_.cycle_state_.cycle.count == 0u) {
    authority_.cycle_state_.retired_cycle = authority_.cycle_state_.cycle.token;
    authority_.cycle_state_.cycle = registry_model::CycleSlot{};
  }
  return true;
}

bool CycleOwner::bind_cycle(const cycle::Flight &flight,
                            std::uint64_t &token) noexcept {
  token = 0u;
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      authority_.execution_state_.slot.token != 0u ||
      retry_ready(authority_.cycle_state_.graph_persists) ||
      authority_.cycle_state_.cycle.token != 0u || flight.token == 0u ||
      flight.bank >= cycle::BankCapacity ||
      flight.bank != flight.ordinal % cycle::BankCapacity ||
      !flight.may_write || !cycle_range(authority_.frames_, flight.input) ||
      !cycle_range(authority_.frames_, flight.output)) {
    return false;
  }
  const auto epoch =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [&flight](const registry_model::LeaseSlot &slot) {
                     return slot.token == flight.token;
                   });
  if (epoch == authority_.cycle_state_.epochs.end() ||
      epoch->state != registry_model::LeaseState::Computing ||
      epoch->cycle != 0u || epoch->cpu_key) {
    return false;
  }
  for (const CacheBinding &binding : epoch->bindings) {
    const cycle::Epoch::Range range =
        binding.access == Access::Read ? flight.input : flight.output;
    if (binding.frame < range.first ||
        binding.frame - range.first >= range.count ||
        (range.mask & (std::uint32_t{1u} << (binding.frame - range.first))) ==
            0u) {
      return false;
    }
  }
  const std::uint64_t cycle_token =
      next_token(authority_.credentials_.next_token);
  if (cycle_token == 0u) {
    return false;
  }
  authority_.cycle_state_.cycle = registry_model::CycleSlot{};
  authority_.cycle_state_.cycle.token = cycle_token;
  authority_.cycle_state_.cycle.tokens[0] = flight.token;
  authority_.cycle_state_.cycle.ordinals[0] = flight.ordinal;
  authority_.cycle_state_.cycle.banks[0] = flight.bank;
  authority_.cycle_state_.cycle.count = 1u;
  epoch->cycle = authority_.cycle_state_.cycle.token;
  token = authority_.cycle_state_.cycle.token;
  return true;
}

bool CycleOwner::advance_cycle(const std::uint64_t token,
                               const cycle::Flight &flight) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() || token == 0u ||
      retry_ready(authority_.cycle_state_.graph_persists) ||
      authority_.cycle_state_.cycle.token != token ||
      authority_.cycle_state_.cycle.closed ||
      authority_.cycle_state_.cycle.count != 1u || flight.token == 0u ||
      flight.bank >= cycle::BankCapacity || !flight.may_write ||
      !cycle_range(authority_.frames_, flight.input) ||
      !cycle_range(authority_.frames_, flight.output)) {
    return false;
  }
  if (flight.ordinal != authority_.cycle_state_.cycle.ordinals[0] + 1u ||
      flight.bank != flight.ordinal % cycle::BankCapacity) {
    return false;
  }
  const auto epoch =
      std::find_if(authority_.cycle_state_.epochs.begin(),
                   authority_.cycle_state_.epochs.end(),
                   [&flight](const registry_model::LeaseSlot &slot) {
                     return slot.token == flight.token;
                   });
  if (flight.token == authority_.cycle_state_.cycle.tokens[0] ||
      epoch == authority_.cycle_state_.epochs.end() ||
      epoch->state != registry_model::LeaseState::Computing ||
      epoch->cycle != 0u || epoch->cpu_key ||
      (flight.ordinal >= cycle::BankCapacity &&
       authority_.cycle_state_.cycle.terminals[flight.bank] !=
           flight.ordinal - cycle::BankCapacity)) {
    return false;
  }
  for (const CacheBinding &binding : epoch->bindings) {
    const cycle::Epoch::Range range =
        binding.access == Access::Read ? flight.input : flight.output;
    if (binding.frame < range.first ||
        binding.frame - range.first >= range.count ||
        (range.mask & (std::uint32_t{1u} << (binding.frame - range.first))) ==
            0u) {
      return false;
    }
  }
  authority_.cycle_state_.cycle.tokens[1] = flight.token;
  authority_.cycle_state_.cycle.ordinals[1] = flight.ordinal;
  authority_.cycle_state_.cycle.banks[1] = flight.bank;
  authority_.cycle_state_.cycle.count = 2u;
  epoch->cycle = token;
  return true;
}

CycleOwner Authority::cycles() noexcept { return CycleOwner{*this}; }

} // namespace rund::compute::detail::residency
