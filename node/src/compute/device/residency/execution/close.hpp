#pragma once

#include "../registry.hpp"
#include "plan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::residency::execution::close_detail {

inline constexpr std::size_t CloseNodeCapacity =
    3u * ExecutionClose::FailureCapacity + WindowCapacity + 2u;

struct CloseWork final {
  std::array<Node, CloseNodeCapacity> nodes{};
  std::size_t count{};
};

[[nodiscard]] inline AuthorityFailure
check_close_frame(const std::vector<Authority::Frame> &frames,
                  const std::uint32_t value, const bool mutate) noexcept {
  if (value >= frames.size()) {
    return AuthorityFailure::Invalid;
  }
  return mutate && frames[value].alias_claims != 0u ? AuthorityFailure::Busy
                                                    : AuthorityFailure::None;
}

[[nodiscard]] inline AuthorityFailure
check_close_region(const std::vector<Authority::Frame> &frames,
                   const FrameRegion region, const bool mutate) noexcept {
  const std::size_t first = region.first;
  const std::size_t count = region.count;
  if (first > frames.size() || count > frames.size() - first) {
    return AuthorityFailure::Invalid;
  }
  for (std::size_t index = first; index < first + count; ++index) {
    const AuthorityFailure result =
        check_close_frame(frames, static_cast<std::uint32_t>(index), mutate);
    if (result != AuthorityFailure::None) {
      return result;
    }
  }
  return AuthorityFailure::None;
}

[[nodiscard]] inline AuthorityFailure
add_close_node(const Plan &plan, const std::vector<Authority::Frame> &frames,
               CloseWork &work, const NodeId id, const bool mutate_output,
               const bool mutate_regions, std::size_t &index) noexcept {
  if (work.count == work.nodes.size()) {
    return AuthorityFailure::Capacity;
  }
  Node node{};
  if (!plan.project(id, node) || node.input_count > UseCapacity ||
      node.output_count > UseCapacity ||
      node.mutation_count > MutationCapacity) {
    return AuthorityFailure::Invalid;
  }
  if (node.id != id) {
    return AuthorityFailure::Invalid;
  }
  AuthorityFailure result =
      check_close_region(frames, node.route.source, false);
  if (result != AuthorityFailure::None) {
    return result;
  }
  result = check_close_region(frames, node.route.target, false);
  if (result != AuthorityFailure::None) {
    return result;
  }
  if (node.output_count > node.route.target.count) {
    return AuthorityFailure::Invalid;
  }
  if (mutate_output) {
    for (std::size_t output = 0u; output < node.output_count; ++output) {
      result = check_close_frame(
          frames, node.route.target.first + static_cast<std::uint32_t>(output),
          true);
      if (result != AuthorityFailure::None) {
        return result;
      }
    }
  }
  if (mutate_regions) {
    for (std::size_t mutation = 0u; mutation < node.mutation_count;
         ++mutation) {
      result = check_close_region(frames, node.mutations[mutation], true);
      if (result != AuthorityFailure::None) {
        return result;
      }
    }
  }
  index = work.count;
  work.nodes[work.count++] = node;
  return AuthorityFailure::None;
}

[[nodiscard]] inline AuthorityFailure
check_close_indices(const std::vector<Authority::Frame> &frames,
                    const std::span<const std::uint32_t> values,
                    const std::size_t count) noexcept {
  if (count > values.size()) {
    return AuthorityFailure::Invalid;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const AuthorityFailure result =
        check_close_frame(frames, values[index], true);
    if (result != AuthorityFailure::None) {
      return result;
    }
  }
  return AuthorityFailure::None;
}

[[nodiscard]] inline AuthorityFailure
check_close_undo(const std::vector<Authority::Frame> &frames,
                 const std::span<const std::uint32_t> undo_frames,
                 const std::size_t undo_count,
                 const std::size_t undo_size) noexcept {
  if (undo_count > undo_frames.size() || undo_count > undo_size) {
    return AuthorityFailure::Invalid;
  }
  for (std::size_t index = 0u; index < undo_count; ++index) {
    const AuthorityFailure result =
        check_close_frame(frames, undo_frames[index], true);
    if (result != AuthorityFailure::None) {
      return result;
    }
  }
  return AuthorityFailure::None;
}

[[nodiscard]] inline AuthorityFailure
check_close_bindings(const std::vector<Authority::Frame> &frames,
                     const std::span<const CacheBinding> bindings,
                     const std::size_t count) noexcept {
  if (count > bindings.size()) {
    return AuthorityFailure::Invalid;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const AuthorityFailure result =
        check_close_frame(frames, bindings[index].frame, true);
    if (result != AuthorityFailure::None) {
      return result;
    }
  }
  return AuthorityFailure::None;
}

} // namespace rund::compute::detail::residency::execution::close_detail
