#include "store.hpp"

#include <algorithm>
#include <limits>

namespace rund::node {

[[nodiscard]] ReactorReadySet* ReactorReadySetFind(
    std::vector<ReactorReadySet>& sets, const ::rund::net::ready::Set handle) noexcept {
  for (ReactorReadySet& set : sets) {
    if (set.live && set.id == handle.id &&
        set.generation == handle.generation) {
      return &set;
    }
  }
  return nullptr;
}

[[nodiscard]] const ReactorReadySet* ReactorReadySetFind(
    const std::vector<ReactorReadySet>& sets,
    const ::rund::net::ready::Set handle) noexcept {
  for (const ReactorReadySet& set : sets) {
    if (set.live && set.id == handle.id &&
        set.generation == handle.generation) {
      return &set;
    }
  }
  return nullptr;
}

[[nodiscard]] std::uint32_t ReactorReadySetLiveCount(
    const std::vector<ReactorReadySet>& sets) noexcept {
  return static_cast<std::uint32_t>(
      std::count_if(sets.begin(), sets.end(),
                    [](const ReactorReadySet& set) { return set.live; }));
}

[[nodiscard]] std::uint32_t ReactorReadySetMemberCount(
    const std::vector<ReactorReadySet>& sets) noexcept {
  std::uint64_t count = 0u;
  for (const ReactorReadySet& set : sets) {
    if (set.live) {
      count += set.members.size();
    }
  }
  return count > std::numeric_limits<std::uint32_t>::max()
             ? std::numeric_limits<std::uint32_t>::max()
             : static_cast<std::uint32_t>(count);
}

[[nodiscard]] bool ReactorReadySetHasDuplicate(
    const ReactorReadySet& set, const ::rund::net::SocketView socket,
    const ReactorInterest interest) noexcept {
  if (!set.live) {
    return false;
  }
  for (const ReactorReadySetMember& member : set.members) {
    if (member.socket == socket && member.interest == interest) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::uint32_t ReactorReadySetClearMembers(
    ReactorReadySet& set) noexcept {
  if (!set.live) {
    set.members.clear();
    return 0u;
  }
  const std::uint32_t removed = static_cast<std::uint32_t>(set.members.size());
  set.members.clear();
  return removed;
}

}  // namespace rund::node
