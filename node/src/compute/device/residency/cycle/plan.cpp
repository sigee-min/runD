#include "plan.hpp"

#include <limits>

namespace rund::compute::detail::residency::cycle {
namespace {

[[nodiscard]] bool
tokens_are_separate(const std::span<const Epoch> epochs) noexcept {
  for (std::size_t left = 0u; left < epochs.size(); ++left) {
    for (const std::uint64_t token : epochs[left].tokens) {
      if (token == 0u) {
        return false;
      }
      for (std::size_t right = left + 1u; right < epochs.size(); ++right) {
        for (const std::uint64_t other : epochs[right].tokens) {
          if (token == other) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool valid_ranges(const Epoch &epoch) noexcept {
  for (std::size_t phase = 0u; phase < PhaseCapacity; ++phase) {
    const Epoch::Range read = epoch.reads[phase];
    const Epoch::Range write = epoch.writes[phase];
    const bool empty = read.count == 0u && write.count == 0u &&
                       read.mask == 0u && write.mask == 0u;
    if (empty) {
      if (phase == index(Phase::Dispatch)) {
        return false;
      }
      continue;
    }
    if (read.count == 0u || write.count == 0u || read.count > 32u ||
        write.count > 32u || read.mask == 0u || write.mask == 0u ||
        (read.count < 32u && (read.mask >> read.count) != 0u) ||
        (write.count < 32u && (write.mask >> write.count) != 0u) ||
        read.first > std::numeric_limits<std::uint32_t>::max() - read.count ||
        write.first > std::numeric_limits<std::uint32_t>::max() - write.count) {
      return false;
    }
  }
  const Epoch::Range upload_read = epoch.reads[index(Phase::Upload)];
  const Epoch::Range upload_write = epoch.writes[index(Phase::Upload)];
  const Epoch::Range dispatch_read = epoch.reads[index(Phase::Dispatch)];
  const Epoch::Range dispatch_write = epoch.writes[index(Phase::Dispatch)];
  const Epoch::Range download_read = epoch.reads[index(Phase::Download)];
  const Epoch::Range download_write = epoch.writes[index(Phase::Download)];
  return upload_read.domain == Domain::Host &&
         (upload_write.domain == Domain::Device ||
          upload_write.domain == Domain::HostVisible) &&
         (upload_write.count == 0u || dispatch_read == upload_write) &&
         (dispatch_write.domain == Domain::Device ||
          dispatch_write.domain == Domain::HostVisible) &&
         download_read == dispatch_write &&
         (download_write.domain == Domain::Host ||
          (download_write.domain == Domain::HostVisible &&
           download_write == download_read)) &&
         !epoch.may_write[index(Phase::Upload)] &&
         epoch.may_write[index(Phase::Dispatch)] &&
         !epoch.may_write[index(Phase::Download)];
}

} // namespace

Result seal(const std::span<const Epoch> epochs) noexcept {
  Result result{};
  if (epochs.empty()) {
    return result;
  }
  if (epochs.size() > EpochCapacity) {
    result.failure = Failure::Capacity;
    return result;
  }
  const std::uint64_t first = epochs.front().ordinal;
  if (first > std::numeric_limits<std::uint64_t>::max() -
                  static_cast<std::uint64_t>(epochs.size() - 1u)) {
    result.failure = Failure::Overflow;
    return result;
  }
  for (std::size_t epoch = 0u; epoch < epochs.size(); ++epoch) {
    const Epoch &input = epochs[epoch];
    if (input.ordinal != first + static_cast<std::uint64_t>(epoch) ||
        input.bank >= BankCapacity ||
        input.bank !=
            static_cast<std::uint32_t>(input.ordinal % BankCapacity) ||
        !valid_ranges(input)) {
      return result;
    }
  }
  if (!tokens_are_separate(epochs)) {
    return result;
  }

  Plan &plan = result.plan;
  for (const Epoch &epoch : epochs) {
    for (std::size_t phase_index = 0u; phase_index < PhaseCapacity;
         ++phase_index) {
      const Phase phase = static_cast<Phase>(phase_index);
      plan.nodes_[plan.node_count_++] = Node{
          .epoch = epoch.ordinal,
          .token = epoch.tokens[phase_index],
          .bank = epoch.bank,
          .phase = phase,
          .source = epoch.reads[phase_index].domain,
          .target = epoch.writes[phase_index].domain,
          .read = epoch.reads[phase_index],
          .write = epoch.writes[phase_index],
          .may_write = epoch.may_write[phase_index],
      };
    }
  }

  for (std::size_t epoch = 0u; epoch < epochs.size(); ++epoch) {
    const std::size_t first_node = epoch * PhaseCapacity;
    plan.edges_[plan.edge_count_++] = Edge{
        .before = static_cast<std::uint8_t>(first_node + index(Phase::Upload)),
        .after = static_cast<std::uint8_t>(first_node + index(Phase::Dispatch)),
    };
    plan.edges_[plan.edge_count_++] = Edge{
        .before =
            static_cast<std::uint8_t>(first_node + index(Phase::Dispatch)),
        .after = static_cast<std::uint8_t>(first_node + index(Phase::Download)),
    };
  }

  // Only the first and third entries can reuse an output bank in the bounded
  // rolling window. Input and Output are distinct physical owners, so the
  // dependency is download(e) -> dispatch(e+2), not upload(e+2). This leaves
  // the next input upload free to overlap the preceding output download.
  if (epochs.size() == EpochCapacity) {
    const std::size_t first_download = index(Phase::Download);
    const std::size_t third_dispatch =
        2u * PhaseCapacity + index(Phase::Dispatch);
    plan.edges_[plan.edge_count_++] = Edge{
        .before = static_cast<std::uint8_t>(first_download),
        .after = static_cast<std::uint8_t>(third_dispatch),
    };
  }
  result.failure = Failure::None;
  return result;
}

} // namespace rund::compute::detail::residency::cycle
