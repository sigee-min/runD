#pragma once

#include <cstdint>
#include <cstddef>

#include "model.hpp"

namespace rund::node {

[[nodiscard]] bool ReactorRegistryPrepare(ReactorRuntime &reactor,
                                          std::size_t capacity) noexcept;
void ReactorRegistryClear(ReactorRuntime &reactor) noexcept;

[[nodiscard]] std::size_t
ReactorRegistrySize(const ReactorRuntime &reactor) noexcept;
[[nodiscard]] bool
ReactorRegistryEmpty(const ReactorRuntime &reactor) noexcept;
[[nodiscard]] const ReactorWait &
ReactorRegistryWaitAt(const ReactorRuntime &reactor, std::size_t index) noexcept;

[[nodiscard]] bool ReactorRegistryAddWait(ReactorRuntime &reactor,
                                          const ReactorWait &wait) noexcept;

[[nodiscard]] const ReactorWait *
ReactorRegistryFindWait(const ReactorRuntime &reactor,
                        std::uint64_t wait_id) noexcept;

[[nodiscard]] const ReactorFdState *
ReactorRegistryFindFd(const ReactorRuntime &reactor,
                      ReactorHandle fd) noexcept;
[[nodiscard]] ReactorFdState *
ReactorRegistryFindFd(ReactorRuntime &reactor, ReactorHandle fd) noexcept;
[[nodiscard]] bool ReactorRegistryEraseFd(ReactorRuntime &reactor,
                                          ReactorHandle fd) noexcept;
[[nodiscard]] std::size_t
ReactorRegistryFdCount(const ReactorRuntime &reactor) noexcept;

[[nodiscard]] std::uint32_t
ReactorRegistryFirstWait(const ReactorRuntime &reactor,
                         ReactorHandle fd) noexcept;
[[nodiscard]] std::uint32_t
ReactorRegistryNextWait(const ReactorRuntime &reactor,
                        std::uint32_t slot) noexcept;
[[nodiscard]] const ReactorWait *
ReactorRegistrySlotWait(const ReactorRuntime &reactor,
                        std::uint32_t slot) noexcept;

[[nodiscard]] bool ReactorRegistryRemoveWait(
    ReactorRuntime &reactor, std::uint64_t wait_id, ReactorWait *removed,
    ReactorInterest *previous_interest = nullptr) noexcept;

[[nodiscard]] bool ReactorRegistryRemoveReadyBatch(
    ReactorRuntime &reactor, const std::vector<ReactorReady> &ordered,
    std::vector<ReactorWait> &removed,
    std::vector<ReactorFdPreviousInterest> &affected) noexcept;

[[nodiscard]] ReactorInterest
ReactorRegistryInterestForFd(const ReactorRuntime &reactor,
                             ReactorHandle fd) noexcept;

[[nodiscard]] bool
ReactorRegistryCollectChangesForWaitAdd(ReactorRuntime &reactor,
                                        const ReactorWait &wait) noexcept;

[[nodiscard]] bool ReactorRegistryCollectChangesForWaitRemove(
    ReactorRuntime &reactor, ReactorHandle fd,
    ReactorInterest previous_interest) noexcept;

} // namespace rund::node
