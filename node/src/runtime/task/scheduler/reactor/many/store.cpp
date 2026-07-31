#include "store.hpp"

#include "../../../../../host/net/interest.hpp"
#include "../../../../../host/net/socket/access.hpp"
#include "../../../../reactor/readiness/handle.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>

namespace rund::node {

[[nodiscard]] ReactorManyGroup *
ReactorManyFindGroup(std::vector<ReactorManyGroup> &groups,
                     const std::uint64_t group_id) noexcept {
  for (ReactorManyGroup &group : groups) {
    if (group.group_id == group_id) {
      return &group;
    }
  }
  return nullptr;
}

[[nodiscard]] const ReactorManyGroup *
ReactorManyFindGroup(const std::vector<ReactorManyGroup> &groups,
                     const std::uint64_t group_id) noexcept {
  for (const ReactorManyGroup &group : groups) {
    if (group.group_id == group_id) {
      return &group;
    }
  }
  return nullptr;
}

[[nodiscard]] std::span<ReactorManyRequest>
ReactorManyRequests(std::vector<ReactorManyRequest> &requests,
                    const ReactorManyGroup &group) noexcept {
  const std::size_t first = group.first_request;
  const std::size_t count = group.request_count;
  if (first > requests.size() || count > requests.size() - first) {
    return {};
  }
  return std::span<ReactorManyRequest>{requests}.subspan(first, count);
}

[[nodiscard]] std::span<const ReactorManyRequest>
ReactorManyRequests(const std::vector<ReactorManyRequest> &requests,
                    const ReactorManyGroup &group) noexcept {
  const std::size_t first = group.first_request;
  const std::size_t count = group.request_count;
  if (first > requests.size() || count > requests.size() - first) {
    return {};
  }
  return std::span<const ReactorManyRequest>{requests}.subspan(first, count);
}

ReasonCode
ReactorManyValidateRequests(const std::span<const ReactorManyRequest> requests,
                            std::vector<std::uint32_t> &index_scratch,
                            std::uint64_t *const comparisons) noexcept {
  if (comparisons == nullptr) {
    return ReasonCode::TaskInvalid;
  }
  *comparisons = 0u;
  try {
    index_scratch.resize(requests.size());
  } catch (...) {
    index_scratch.clear();
    return ReasonCode::ReactorWaitCapacityExceeded;
  }
  std::iota(index_scratch.begin(), index_scratch.end(), 0u);
  for (const ReactorManyRequest &request : requests) {
    if (request.fd == kInvalidReactorHandle) {
      return ReasonCode::IoFdInvalid;
    }
    if (request.interest == ReactorInterest::None) {
      return ReasonCode::TaskInvalid;
    }
  }
  const auto less = [&requests, comparisons](const std::uint32_t left,
                                             const std::uint32_t right) {
    ++(*comparisons);
    const ReactorManyRequest &a = requests[left];
    const ReactorManyRequest &b = requests[right];
    if (a.fd != b.fd) {
      return a.fd < b.fd;
    }
    const std::uint64_t a_generation =
        ::rund::net::detail::SocketAccess::generation(a.socket);
    const std::uint64_t b_generation =
        ::rund::net::detail::SocketAccess::generation(b.socket);
    if (a_generation != b_generation) {
      return a_generation < b_generation;
    }
    return static_cast<std::uint8_t>(a.interest) <
           static_cast<std::uint8_t>(b.interest);
  };
  std::sort(index_scratch.begin(), index_scratch.end(), less);
  for (std::size_t index = 1u; index < index_scratch.size(); ++index) {
    ++(*comparisons);
    const ReactorManyRequest &previous = requests[index_scratch[index - 1u]];
    const ReactorManyRequest &current = requests[index_scratch[index]];
    if (previous.fd == current.fd && previous.socket == current.socket &&
        previous.interest == current.interest) {
      return ReasonCode::TaskInvalid;
    }
  }
  return ReasonCode::Ok;
}

bool ReactorManyBuildRequests(
    const std::span<const ::rund::net::ready::Request> requests,
    const std::span<const ::rund::net::SocketLease> leases,
    std::vector<ReactorManyRequest> &out) noexcept {
  if (leases.size() != requests.size()) {
    out.clear();
    return false;
  }
  try {
    out.clear();
    out.resize(requests.size());
  } catch (...) {
    out.clear();
    return false;
  }
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    out[index] = ReactorManyRequest{
        .socket = requests[index].socket,
        .fd = ReactorHandleFromPublic(leases[index].native()),
        .slot = static_cast<std::uint32_t>(index),
        .event_index = static_cast<std::uint32_t>(index),
        .interest = ::rund::net::ReactorInterestFor(requests[index].interest),
    };
  }
  return true;
}

[[nodiscard]] const ReactorManyRequest *
ReactorManyFindRequest(const std::span<const ReactorManyRequest> requests,
                       const std::uint64_t wait_id) noexcept {
  if (requests.empty() || wait_id < requests.front().wait_id) {
    return nullptr;
  }
  const std::uint64_t offset = wait_id - requests.front().wait_id;
  if (offset >= requests.size()) {
    return nullptr;
  }
  const ReactorManyRequest &request =
      requests[static_cast<std::size_t>(offset)];
  return request.wait_id == wait_id ? &request : nullptr;
}

void ReactorManyEraseGroup(std::vector<ReactorManyGroup> &groups,
                           std::vector<ReactorManyRequest> &requests,
                           const std::uint64_t group_id) noexcept {
  const auto group_it = std::find_if(groups.begin(), groups.end(),
                                     [group_id](const ReactorManyGroup &group) {
                                       return group.group_id == group_id;
                                     });
  if (group_it == groups.end()) {
    return;
  }

  const ReactorManyGroup erased = *group_it;
  groups.erase(group_it);

  std::size_t removed = 0u;
  const std::size_t first = erased.first_request;
  if (first < requests.size()) {
    const std::size_t available = requests.size() - first;
    const std::size_t count =
        std::min<std::size_t>(erased.request_count, available);
    std::size_t last = first;
    const std::size_t limit = first + count;
    while (last < limit && requests[last].group_id == group_id) {
      ++last;
    }
    removed = last - first;
    if (removed != 0u) {
      requests.erase(requests.begin() + static_cast<std::ptrdiff_t>(first),
                     requests.begin() +
                         static_cast<std::ptrdiff_t>(first + removed));
    }
  }

  if (removed != 0u) {
    const std::size_t erased_end = first + removed;
    for (ReactorManyGroup &group : groups) {
      if (group.first_request >= erased_end) {
        group.first_request =
            group.first_request > removed
                ? static_cast<std::uint32_t>(group.first_request - removed)
                : 0u;
      }
    }
  }
}

} // namespace rund::node
