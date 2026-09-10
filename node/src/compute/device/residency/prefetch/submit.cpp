#include "local.hpp"

#include "../registry.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail::residency {

bool Prefetcher::submit(VirtualBacking &backing,
                        const std::span<const PrefetchRequest> requests,
                        const std::uint64_t token, const bool speculative,
                        const bool coherent_input, const bool coherent_deferred,
                        const std::span<AliasLease> aliases) noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Idle || requests.empty() ||
      requests.size() > requests_.size() || token == 0u || alias_count_ != 0u ||
      aliases.size() > aliases_.size()) {
    return false;
  }
  const auto find_alias = [&](const PrefetchRequest &request) {
    return std::find_if(
        aliases.begin(), aliases.end(), [&](const AliasLease &alias) {
          return alias &&
                 prefetch_detail::alias_request_matches(alias, request, token);
        });
  };
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (requests[index].key.backing == 0u || requests[index].bytes == 0u ||
        requests[index].frame == nullptr ||
        requests[index].target_offset > page_bytes_ ||
        requests[index].bytes > page_bytes_ - requests[index].target_offset ||
        (requests[index].alias_reuse &&
         (requests[index].alias_frame == nullptr ||
          requests[index].alias_nonce == 0u ||
          requests[index].alias_generation == 0u ||
          requests[index].alias_owner_token != token ||
          find_alias(requests[index]) == aliases.end())) ||
        (requests[index].fetch &&
         (requests[index].read_target_offset > page_bytes_ ||
          requests[index].read_bytes >
              page_bytes_ - requests[index].read_target_offset ||
          (requests[index].reuse_bytes != 0u &&
           ((!requests[index].alias_reuse && index == 0u) ||
            requests[index].reuse_source_offset > page_bytes_ ||
            requests[index].reuse_bytes >
                page_bytes_ - requests[index].reuse_source_offset ||
            requests[index].reuse_target_offset > page_bytes_ ||
            requests[index].reuse_bytes >
                page_bytes_ - requests[index].reuse_target_offset))))) {
      return false;
    }
    requests_[index] = requests[index];
    pages_[index] = PrefetchedPage{
        .key = requests[index].key,
        .bytes = requests[index].bytes,
        .backing_bytes = 0u,
        .target_offset = requests[index].target_offset,
        .frame = requests[index].frame,
        .physical_frame = requests[index].physical_frame,
        .fetched = requests[index].fetch,
    };
  }
  for (std::size_t index = 0u; index < aliases.size(); ++index) {
    aliases_[index] = std::move(aliases[index]);
  }
  alias_count_ = aliases.size();
  backing_ = &backing;
  count_ = requests.size();
  token_ = token;
  speculative_ = speculative;
  coherent_input_ = coherent_input;
  coherent_deferred_ = coherent_deferred;
  status_ = Status::success();
  io_ns_ = 0u;
  state_ = State::Pending;
  pending_.notify_one();
  return true;
}

} // namespace rund::compute::detail::residency
