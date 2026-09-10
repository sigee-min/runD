#include "../../registry.hpp"
#include "../internal.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::residency::graph_epoch_detail {

AuthorityResult Assignment::run(
    const Authority &authority,
    const std::span<const GraphPortRequest> ports, const std::uint64_t epoch,
    AdmissionDraft &draft) noexcept {
  for (const GraphPortRequest &port : ports) {
    const registry_model::Frame &selected =
        authority.frames_[port.region.first];
    if (selected.extent == 0u) {
      continue;
    }
    const bool conflicting_view = std::any_of(
        authority.frames_.begin(), authority.frames_.end(),
        [&selected](const registry_model::Frame &frame) {
          return frame.assigned && frame.extent == selected.extent &&
                 frame.view != selected.view && frame.state != FrameState::Empty;
        });
    if (conflicting_view) {
      return AuthorityResult{.failure = AuthorityFailure::Capacity};
    }
  }

  draft.existing_frames.fill(NoFrame);
  for (std::size_t port_index = 0u; port_index < ports.size(); ++port_index) {
    const GraphPortRequest &request = ports[port_index];
    for (std::size_t page = 0u; page < draft.page_count; ++page) {
      const CacheUse &use = draft.projected[request.first_use + page];
      std::uint32_t existing = NoFrame;
      for (std::size_t cache_index = 0u;
           cache_index < request.cache_region_count; ++cache_index) {
        const FrameRegion cache = request.cache_regions[cache_index];
        const std::size_t found = find_frame(
            authority.frames_, use.key, cache.first, cache.count);
        if (found == authority.frames_.size()) {
          continue;
        }
        if (existing != NoFrame ||
            found > std::numeric_limits<std::uint32_t>::max()) {
          return AuthorityResult{.failure = AuthorityFailure::Invalid};
        }
        existing = static_cast<std::uint32_t>(found);
      }
      draft.existing_frames[request.first_use + page] = existing;
    }
  }

  draft.local_count = ports.front().region.count;
  const Score replacement_bound = static_cast<Score>(
      draft.page_count * ports.size() * (draft.local_count + 1u) + 1u);
  for (std::size_t page = 0u; page < draft.page_count; ++page) {
    bool any = false;
    for (std::size_t local = 0u; local < draft.local_count; ++local) {
      bool feasible = true;
      Score hits = 0;
      Score replacement = 0;
      for (std::size_t port_index = 0u; port_index < ports.size();
           ++port_index) {
        const GraphPortRequest &request = ports[port_index];
        const std::span<const CacheUse> port_uses{
            draft.projected.data() + request.first_use, request.use_count};
        const CacheUse &use = port_uses[page];
        const std::size_t target = request.region.first + local;
        const std::uint32_t existing =
            draft.existing_frames[request.first_use + page];
        const registry_model::Frame &frame = authority.frames_[target];
        if (frame.state == FrameState::Mapping ||
            frame.state == FrameState::Pinned ||
            frame.state == FrameState::Writeback) {
          feasible = false;
          break;
        }
        if (existing == target) {
          if (writes(use.access) && !frame.dirty.empty()) {
            feasible = false;
            break;
          }
          hits += 2;
          continue;
        }
        if (existing != NoFrame) {
          const registry_model::Frame &source = authority.frames_[existing];
          if (!reads(use.access) ||
              (source.state != FrameState::Resident &&
               source.state != FrameState::Dirty) ||
              (use.key.domain == CacheDomain::Backing &&
               !source.dirty.empty())) {
            feasible = false;
            break;
          }
          ++hits;
        } else if (reads(use.access) &&
                   use.key.domain == CacheDomain::Transient) {
          feasible = false;
          break;
        }
        const bool demanded_source = std::any_of(
            draft.existing_frames.begin() +
                static_cast<std::ptrdiff_t>(request.first_use),
            draft.existing_frames.begin() + static_cast<std::ptrdiff_t>(
                request.first_use + draft.page_count),
            [target](const std::uint32_t candidate) {
              return candidate == target;
            });
        if (!demanded_source && frame.state != FrameState::Empty &&
            (demanded(port_uses, frame.key) ||
             (frame.retain_until != NeverUse && epoch <= frame.retain_until) ||
             !frame.dirty.empty())) {
          feasible = false;
          break;
        }
        if (existing != NoFrame) {
          replacement += static_cast<Score>(draft.local_count);
        } else if (frame.state == FrameState::Empty) {
          replacement += static_cast<Score>(draft.local_count + 1u);
        } else {
          Score rank = 0;
          for (std::size_t candidate = request.region.first;
               candidate < static_cast<std::size_t>(request.region.first) +
                               request.region.count;
               ++candidate) {
            const registry_model::Frame &other = authority.frames_[candidate];
            if (other.state != FrameState::Empty &&
                (other.next_use < frame.next_use ||
                 (other.next_use == frame.next_use && other.key < frame.key))) {
              ++rank;
            }
          }
          replacement += rank;
        }
      }
      if (!feasible) {
        draft.costs[page][local] = Forbidden;
        continue;
      }
      const Score score = hits * replacement_bound + replacement;
      draft.maximum_score = std::max(draft.maximum_score, score);
      draft.costs[page][local] = score;
      any = true;
    }
    if (!any) {
      return AuthorityResult{.failure = AuthorityFailure::Capacity};
    }
  }
  for (std::size_t page = 0u; page < draft.page_count; ++page) {
    for (std::size_t local = 0u; local < draft.local_count; ++local) {
      if (draft.costs[page][local] != Forbidden) {
        draft.costs[page][local] =
            draft.maximum_score - draft.costs[page][local];
      }
    }
  }

  // Rectangular Hungarian assignment: pages are rows, physical locals are
  // columns. The cost construction first maximizes total hits, then the
  // deterministic farthest-next-use/empty-frame replacement preference.
  std::array<Score, PipelineLeafCapacity + 1u> row_potential{};
  std::array<Score, PipelineLeafCapacity + 1u> column_potential{};
  std::array<std::size_t, PipelineLeafCapacity + 1u> column_row{};
  std::array<std::size_t, PipelineLeafCapacity + 1u> path{};
  for (std::size_t row = 1u; row <= draft.page_count; ++row) {
    column_row[0] = row;
    std::size_t column0 = 0u;
    std::array<Score, PipelineLeafCapacity + 1u> minimum{};
    minimum.fill(Forbidden);
    std::array<bool, PipelineLeafCapacity + 1u> used{};
    do {
      used[column0] = true;
      const std::size_t row0 = column_row[column0];
      Score delta = Forbidden;
      std::size_t column1 = 0u;
      for (std::size_t column = 1u; column <= draft.local_count; ++column) {
        if (used[column]) {
          continue;
        }
        const Score raw = draft.costs[row0 - 1u][column - 1u];
        if (raw == Forbidden) {
          continue;
        }
        const Score current =
            raw - row_potential[row0] - column_potential[column];
        if (current < minimum[column]) {
          minimum[column] = current;
          path[column] = column0;
        }
        if (minimum[column] < delta ||
            (minimum[column] == delta && column < column1)) {
          delta = minimum[column];
          column1 = column;
        }
      }
      if (delta == Forbidden || column1 == 0u) {
        return AuthorityResult{.failure = AuthorityFailure::Capacity};
      }
      for (std::size_t column = 0u; column <= draft.local_count; ++column) {
        if (used[column]) {
          row_potential[column_row[column]] += delta;
          column_potential[column] -= delta;
        } else if (minimum[column] != Forbidden) {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (column_row[column0] != 0u);
    do {
      const std::size_t prior = path[column0];
      column_row[column0] = column_row[prior];
      column0 = prior;
    } while (column0 != 0u);
  }
  for (std::size_t column = 1u; column <= draft.local_count; ++column) {
    if (column_row[column] != 0u) {
      draft.locals[column_row[column] - 1u] =
          static_cast<std::uint32_t>(column - 1u);
    }
  }
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency::graph_epoch_detail
