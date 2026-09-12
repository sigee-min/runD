#include "internal.hpp"

#include <algorithm>

namespace rund::compute::detail::residency {

AuthorityResult GraphPromoteOwner::validate_graph_promote_group(
    const std::span<execution::GraphReady> ready,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint64_t destination_token,
    graph_promote_detail::Group &group) const noexcept {
  group.owner.reset();
  group.source_count = 0u;
  group.tokens.fill(0u);
  if (ready.empty() || ready.size() > execution::GraphPromoteSourceCapacity ||
      destination_token == 0u) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  group.owner = ready.front().plan_owner_;
  if (!graph_promote_detail::project_group(group.owner, invocation, batch,
                                           stage, group.projection)) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  for (std::size_t index = 0u; index < ready.size(); ++index) {
    const execution::GraphReady &source = ready[index];
    const auto pages = source.pages();
    if (!source || source.owner_ != &authority_ || group.owner == nullptr ||
        source.plan_owner_.get() != group.owner.get() || pages.empty()) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    const std::uint32_t resource = pages.front().use.resource;
    if (resource == 0u ||
        std::any_of(pages.begin(), pages.end(),
                    [resource](const auto page) {
                      return page.use.resource != resource;
                    }) ||
        std::find(group.resources.begin(), group.resources.begin() + index,
                  resource) != group.resources.begin() + index ||
        !graph_promote_detail::validate_source(
            group.owner, source, stage, resource, group.projection,
            group.first_uses[index])) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    group.resources[index] = resource;
    group.pages[index] = pages;
  }
  group.source_count = ready.size();
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency
