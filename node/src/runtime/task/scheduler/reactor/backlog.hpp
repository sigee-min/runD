#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node {

struct ReactorReady;
struct ReactorRuntime;
[[nodiscard]] bool
ReactorBacklogHasReady(const ReactorRuntime &reactor) noexcept;

[[nodiscard]] bool ReactorBacklogStoreSuffix(
    ReactorRuntime &reactor, ::rund::detail::task::StatStorage &stats,
    const std::vector<ReactorReady> &ordered, std::size_t consumed) noexcept;

[[nodiscard]] bool ReactorBacklogTakePrefix(
    ReactorRuntime &reactor, ::rund::detail::task::StatStorage &stats,
    std::size_t budget, std::vector<ReactorReady> &out) noexcept;

void ReactorBacklogClear(ReactorRuntime &reactor) noexcept;

void ReactorBacklogRemoveFd(ReactorRuntime &reactor, int fd) noexcept;
void ReactorBacklogRemoveWait(ReactorRuntime &reactor,
                              std::uint64_t wait_id) noexcept;

} // namespace rund::node
