#include "internal.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <mutex>

namespace rund::compute::detail {

void publish_private_pipeline_terminal_pair(
    PipelineState &first, const PipelineTerminal first_terminal,
    PipelineState &second, const PipelineTerminal second_terminal,
    void *const publication, void (*const commit)(void *) noexcept) noexcept {
  std::array<PipelineState *, 2u> pipelines{&first, &second};
  std::array<PipelineTerminal, 2u> terminals{first_terminal, second_terminal};
  publish_private_pipeline_terminal_set(pipelines, terminals, publication,
                                        commit);
}

void publish_private_pipeline_terminal_set(
    const std::span<PipelineState *const> pipelines,
    const std::span<const PipelineTerminal> terminals, void *const publication,
    void (*const commit)(void *) noexcept) noexcept {
  constexpr std::size_t Capacity = residency::TiledGraphResourceCapacity;
  if (pipelines.empty() || pipelines.size() > Capacity ||
      pipelines.size() != terminals.size() || publication == nullptr ||
      commit == nullptr) {
    return;
  }
  std::array<PipelinePublicationState *, Capacity> ordered{};
  std::size_t publication_count = 0u;
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    PipelineState *const pipeline = pipelines[index];
    if (pipeline == nullptr || pipeline->publication == nullptr ||
        std::find(pipelines.begin(), pipelines.begin() + index, pipeline) !=
            pipelines.begin() + index) {
      return;
    }
    PipelinePublicationState *const candidate = pipeline->publication.get();
    if (std::find(ordered.begin(), ordered.begin() + publication_count,
                  candidate) != ordered.begin() + publication_count) {
      return;
    }
    ordered[publication_count++] = candidate;
  }
  std::sort(ordered.begin(), ordered.begin() + publication_count,
            std::less<PipelinePublicationState *>{});
  std::array<std::unique_lock<std::mutex>, Capacity> locks{};
  for (std::size_t index = 0u; index < publication_count; ++index) {
    locks[index] = std::unique_lock<std::mutex>{ordered[index]->gate};
  }
  // The Pipeline gates are held by the caller. Preflight the complete cohort
  // while publication identities are stable, so one bad member cannot leave a
  // partially closed set or invoke the backing callback.
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    if (!claim_detail::private_terminal_check(*pipelines[index],
                                              terminals[index], true)
             .valid) {
      return;
    }
  }
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    publish_pipeline_terminal_locked(*pipelines[index], terminals[index],
                                     PipelineClaimAuthority::PrivateResidency,
                                     true);
  }
  commit(publication);
}

void publish_shared_pipeline_terminal_pair(
    PipelineState &first, const PipelineTerminal first_terminal,
    PipelineState &second, const PipelineTerminal second_terminal,
    void *const publication, void (*const commit)(void *) noexcept) noexcept {
  PipelinePublicationState *const first_publication = first.publication.get();
  PipelinePublicationState *const second_publication = second.publication.get();
  if (first.device == nullptr || first.device != second.device ||
      first.device->claims == nullptr || first_publication == nullptr ||
      second_publication == nullptr || commit == nullptr) {
    return;
  }
  const auto publish = [&]() noexcept {
    std::lock_guard claim_lock{first.device->claims->gate};
    publish_pipeline_terminal_locked(
        first, first_terminal, PipelineClaimAuthority::Shared, false, true);
    publish_pipeline_terminal_locked(
        second, second_terminal, PipelineClaimAuthority::Shared, false, true);
    commit(publication);
  };
  if (first_publication == second_publication) {
    std::lock_guard publication_lock{first_publication->gate};
    publish();
    return;
  }
  if (std::less<PipelinePublicationState *>{}(first_publication,
                                              second_publication)) {
    std::scoped_lock publication_locks{first_publication->gate,
                                       second_publication->gate};
    publish();
    return;
  }
  std::scoped_lock publication_locks{second_publication->gate,
                                     first_publication->gate};
  publish();
}

} // namespace rund::compute::detail
