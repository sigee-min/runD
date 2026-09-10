#include "../../../registry.hpp"

#include "../../../execution/plan.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency {

bool Authority::validate_execution_regions_locked(
    const execution::Plan &plan, const bool sliding,
    registry_model::ExecutionSlot &candidate) noexcept {
  const auto reserve = [this, &candidate,
                        sliding](const FrameRegion region) noexcept {
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    if (count == 0u || first > frames_.size() ||
        count > frames_.size() - first) {
      return false;
    }
    for (std::size_t index = first; index < first + count; ++index) {
      const Frame &frame = frames_[index];
      if (!frame.assigned || frame.tier != region.tier ||
          frame.role != region.role ||
          (sliding &&
           (frame.state == FrameState::Mapping ||
            frame.state == FrameState::Pinned ||
            frame.state == FrameState::Writeback || !frame.dirty.empty()))) {
        return false;
      }
      const std::uint32_t value = static_cast<std::uint32_t>(index);
      const auto prior =
          std::find(candidate.frames.begin(),
                    candidate.frames.begin() +
                        static_cast<std::ptrdiff_t>(candidate.frame_count),
                    value);
      if (prior != candidate.frames.begin() +
                       static_cast<std::ptrdiff_t>(candidate.frame_count)) {
        continue;
      }
      if (candidate.frame_count == candidate.frames.size()) {
        return false;
      }
      candidate.frames[candidate.frame_count] = value;
      ++candidate.frame_count;
    }
    return true;
  };

  std::array<FrameRegion, registry_model::ExecutionRegionCapacity> owners{};
  std::size_t owner_count = 0u;
  const std::size_t bank_count =
      plan.epoch_count() < execution::BankCapacity
          ? static_cast<std::size_t>(plan.epoch_count())
          : execution::BankCapacity;
  for (std::size_t bank = 0u; bank < bank_count; ++bank) {
    execution::Node input{};
    execution::Node dispatch{};
    execution::Node output{};
    if (!plan.project(
            execution::NodeId{.epoch = bank, .phase = execution::Phase::Input},
            input) ||
        !plan.project(execution::NodeId{.epoch = bank,
                                        .phase = execution::Phase::Dispatch},
                      dispatch) ||
        !plan.project(
            execution::NodeId{.epoch = bank, .phase = execution::Phase::Output},
            output)) {
      return false;
    }
    owners[owner_count++] = input.route.source;
    owners[owner_count++] = input.route.target;
    owners[owner_count++] = dispatch.route.target;
    owners[owner_count++] = output.route.target;
  }
  for (std::size_t left = 0u; left < owner_count; ++left) {
    const FrameRegion region = owners[left];
    const std::size_t end =
        static_cast<std::size_t>(region.first) + region.count;
    if (!reserve(region)) {
      return false;
    }
    for (std::size_t right = left + 1u; right < owner_count; ++right) {
      const FrameRegion other = owners[right];
      const std::size_t other_end =
          static_cast<std::size_t>(other.first) + other.count;
      if (!(end <= other.first || other_end <= region.first)) {
        return false;
      }
    }
  }

  std::array<std::uint64_t, 3u> representatives{
      0u,
      plan.epoch_count() > 1u ? 1u : 0u,
      plan.epoch_count() - 1u,
  };
  for (std::size_t representative = 0u; representative < representatives.size();
       ++representative) {
    const std::uint64_t epoch = representatives[representative];
    if (representative != 0u &&
        std::find(representatives.begin(),
                  representatives.begin() +
                      static_cast<std::ptrdiff_t>(representative),
                  epoch) != representatives.begin() +
                                static_cast<std::ptrdiff_t>(representative)) {
      continue;
    }
    for (std::size_t phase = 0u; phase < 3u; ++phase) {
      execution::Node node{};
      const execution::NodeId id{
          .epoch = epoch,
          .phase = static_cast<execution::Phase>(phase),
      };
      if (!plan.project(id, node) || node.id != id ||
          node.bank != epoch % execution::BankCapacity ||
          node.input_count > execution::UseCapacity ||
          node.output_count > execution::UseCapacity ||
          node.mutation_count == 0u ||
          node.mutation_count > execution::MutationCapacity ||
          !reserve(node.route.source) || !reserve(node.route.target)) {
        return false;
      }
      for (std::size_t index = 0u; index < node.input_count; ++index) {
        const CacheUse &use = node.input[index];
        if (use.key.backing == 0u || use.access != Access::Read ||
            !use.dirty.empty() || use.epoch != epoch ||
            (use.retain_until != NeverUse && use.retain_until < epoch)) {
          return false;
        }
      }
      for (std::size_t index = 0u; index < node.output_count; ++index) {
        const CacheUse &use = node.output[index];
        if (use.key.backing == 0u || use.access != Access::Write ||
            use.dirty.empty() || use.epoch != epoch ||
            (use.retain_until != NeverUse && use.retain_until < epoch)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::residency
