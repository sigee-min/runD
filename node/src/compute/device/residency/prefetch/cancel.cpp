#include "local.hpp"

#include "../registry.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail::residency {

bool Prefetcher::release_aliases_locked(Authority &authority,
                                        const bool source_known) noexcept {
  if (state_ != State::Idle) {
    return false;
  }
  for (std::size_t alias_index = 0u; alias_index < alias_count_;
       ++alias_index) {
    const AliasLease &alias = aliases_[alias_index];
    if (!alias) {
      continue;
    }
    const auto request =
        std::find_if(requests_.begin(), requests_.begin() + count_,
                     [&alias](const PrefetchRequest &candidate) {
                       return candidate.alias_reuse &&
                              prefetch_detail::alias_request_matches(
                                  alias, candidate, alias.owner_token());
                     });
    if (request == requests_.begin() + count_) {
      return false;
    }
  }
  bool clean = true;
  for (std::size_t index = 0u; index < alias_count_; ++index) {
    if (aliases_[index] &&
        !authority.release_alias(std::move(aliases_[index]), source_known)) {
      clean = false;
    }
  }
  if (clean) {
    alias_count_ = 0u;
  }
  return clean;
}

bool Prefetcher::cancel_locked(Authority &authority, const std::uint64_t token,
                               const bool source_known,
                               const bool invalidate) noexcept {
  if (state_ == State::Ready) {
    backing_ = nullptr;
    speculative_ = false;
    coherent_input_ = false;
    coherent_deferred_ = false;
    state_ = State::Idle;
  }
  if (state_ != State::Idle) {
    return false;
  }
  if (token != 0u) {
    if (token_ != 0u && token_ != token) {
      return false;
    }
    token_ = token;
  }
  if (!release_aliases_locked(authority, source_known)) {
    return false;
  }
  if (token_ == 0u) {
    return true;
  }
  if (!authority.complete(token_, false, invalidate)) {
    return false;
  }
  token_ = 0u;
  speculative_ = false;
  coherent_input_ = false;
  coherent_deferred_ = false;
  return true;
}

bool Prefetcher::cancel(Authority &authority, const bool source_known,
                        const bool invalidate) noexcept {
  std::unique_lock lock{gate_};
  if (state_ == State::Pending) {
    ready_.wait(lock, [this] { return state_ != State::Pending; });
  }
  return cancel_locked(authority, token_, source_known, invalidate);
}

bool Prefetcher::cancel(Authority &authority, PrefetchReceipt &receipt,
                        const bool source_known,
                        const bool invalidate) noexcept {
  std::unique_lock lock{gate_};
  if (state_ == State::Pending) {
    ready_.wait(lock, [this] { return state_ != State::Pending; });
  }
  if (receipt.token != 0u) {
    if (token_ != 0u && token_ != receipt.token) {
      return false;
    }
    token_ = receipt.token;
  }
  const bool clean =
      cancel_locked(authority, receipt.token, source_known, invalidate);
  if (clean) {
    receipt.token = 0u;
  }
  return clean;
}

bool Prefetcher::release_aliases(Authority &authority,
                                 const bool source_known) noexcept {
  std::lock_guard lock{gate_};
  return release_aliases_locked(authority, source_known);
}

} // namespace rund::compute::detail::residency
