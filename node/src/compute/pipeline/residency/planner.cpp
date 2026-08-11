#include "planner.hpp"

#include "identity.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace rund::compute::detail::residency {
namespace {

struct Frame final {
  PageKey key{};
  bool occupied{};
  bool dirty{};
};

[[nodiscard]] Access merge_access(const Access left,
                                  const Access right) noexcept {
  if (left == right) {
    return left;
  }
  return Access::ReadWrite;
}

void canonicalize(std::vector<PageUse> &uses) {
  std::sort(uses.begin(), uses.end(),
            [](const PageUse &left, const PageUse &right) {
              return left.key < right.key;
            });
  std::size_t written = 0u;
  for (const PageUse use : uses) {
    if (written != 0u && uses[written - 1u].key == use.key) {
      uses[written - 1u].access =
          merge_access(uses[written - 1u].access, use.access);
    } else {
      uses[written++] = use;
    }
  }
  uses.resize(written);
}

[[nodiscard]] bool demanded(const DemandEpoch &epoch,
                            const PageKey key) noexcept {
  const auto found =
      std::lower_bound(epoch.uses.begin(), epoch.uses.end(), key,
                       [](const PageUse &use, const PageKey candidate) {
                         return use.key < candidate;
                       });
  return found != epoch.uses.end() && found->key == key;
}

[[nodiscard]] std::uint64_t next_use(const GraphPlanInput &input,
                                     const std::size_t after,
                                     const PageKey key) noexcept {
  for (std::size_t index = after; index < input.epochs.size(); ++index) {
    if (demanded(input.epochs[index], key)) {
      return index;
    }
  }
  return std::numeric_limits<std::uint64_t>::max();
}

[[nodiscard]] std::size_t find_frame(const std::vector<Frame> &frames,
                                     const PageKey key) noexcept {
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    if (frames[index].occupied && frames[index].key == key) {
      return index;
    }
  }
  return frames.size();
}

[[nodiscard]] std::size_t
select_frame(const GraphPlanInput &input, const std::size_t epoch_index,
             const std::vector<Frame> &frames) noexcept {
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    if (!frames[index].occupied) {
      return index;
    }
  }
  std::size_t selected = frames.size();
  std::uint64_t farthest = 0u;
  for (std::size_t index = 0u; index < frames.size(); ++index) {
    const Frame &frame = frames[index];
    if (demanded(input.epochs[epoch_index], frame.key)) {
      continue;
    }
    const std::uint64_t reuse = next_use(input, epoch_index + 1u, frame.key);
    if (selected == frames.size() || reuse > farthest ||
        (reuse == farthest && frames[selected].key < frame.key)) {
      selected = index;
      farthest = reuse;
    }
  }
  return selected;
}

} // namespace

PlanResult PlanResidency(const StreamPlanInput &input) noexcept {
  if (input.page_bytes == 0u) {
    return PlanResult{.failure = Failure::Invalid};
  }
  if (input.requested_frames > input.max_frames ||
      input.max_frames > std::numeric_limits<std::uint32_t>::max()) {
    return PlanResult{.failure = Failure::Capacity};
  }
  if (input.page_count != 0u && input.requested_frames == 0u) {
    return PlanResult{.failure = Failure::Infeasible};
  }
  const std::uint64_t frames =
      std::min(input.page_count, input.requested_frames);
  const StreamPlan stream{input.page_count, frames};
  const Identity identity =
      IdentifyResidencyPlan(input.page_bytes, input.page_count, frames);
  return PlanResult{
      .failure = Failure::None,
      .plan = ResidencyPlan{input.page_bytes, stream, identity},
  };
}

PlanResult PlanResidency(const GraphPlanInput &source) noexcept {
  if (source.page_bytes == 0u || source.frame_capacity == 0u ||
      source.epochs.empty()) {
    return PlanResult{.failure = Failure::Invalid};
  }
  try {
    GraphPlanInput input = source;
    for (DemandEpoch &epoch : input.epochs) {
      canonicalize(epoch.uses);
      if (epoch.uses.empty() || epoch.uses.size() > input.frame_capacity) {
        return PlanResult{.failure = Failure::Infeasible};
      }
    }

    std::vector<Frame> frames(input.frame_capacity);
    std::vector<PageUse> uses;
    std::vector<Transition> transitions;
    std::vector<Epoch> epochs;
    std::size_t total_uses = 0u;
    for (const DemandEpoch &epoch : input.epochs) {
      total_uses += epoch.uses.size();
    }
    uses.reserve(total_uses);
    transitions.reserve(total_uses * 4u);
    epochs.reserve(input.epochs.size());

    for (std::size_t epoch_index = 0u; epoch_index < input.epochs.size();
         ++epoch_index) {
      const DemandEpoch &demand = input.epochs[epoch_index];
      Epoch epoch{
          .node = demand.node,
          .tile = demand.tile,
          .first_use = uses.size(),
          .use_count = demand.uses.size(),
          .first_transition = transitions.size(),
      };
      uses.insert(uses.end(), demand.uses.begin(), demand.uses.end());
      for (const PageUse use : demand.uses) {
        std::size_t frame = find_frame(frames, use.key);
        if (frame == frames.size()) {
          frame = select_frame(input, epoch_index, frames);
          if (frame == frames.size()) {
            return PlanResult{.failure = Failure::Infeasible};
          }
          if (frames[frame].occupied) {
            if (frames[frame].dirty) {
              transitions.push_back(Transition{
                  .key = frames[frame].key,
                  .frame = static_cast<std::uint32_t>(frame),
                  .kind = TransitionKind::Writeback,
              });
            }
            transitions.push_back(Transition{
                .key = frames[frame].key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = TransitionKind::Unmap,
            });
          }
          transitions.push_back(Transition{
              .key = use.key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Fetch,
          });
          transitions.push_back(Transition{
              .key = use.key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Map,
          });
          frames[frame] = Frame{.key = use.key, .occupied = true};
        }
        frames[frame].dirty = frames[frame].dirty || writes(use.access);
      }
      epoch.transition_count = transitions.size() - epoch.first_transition;
      epochs.push_back(epoch);
    }

    const Identity identity = IdentifyResidencyPlan(
        input.page_bytes, input.frame_capacity, uses, epochs);
    return PlanResult{
        .failure = Failure::None,
        .plan = ResidencyPlan{input.page_bytes, input.frame_capacity,
                              std::move(uses), std::move(transitions),
                              std::move(epochs), identity},
    };
  } catch (const std::bad_alloc &) {
    return PlanResult{.failure = Failure::Capacity};
  }
}

} // namespace rund::compute::detail::residency
