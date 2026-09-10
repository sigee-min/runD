#pragma once

#include "../registry.hpp"

namespace rund::node::reactor_registry_detail {

using OrderIterator = std::vector<std::uint32_t>::iterator;
using ConstOrderIterator = std::vector<std::uint32_t>::const_iterator;
using FdIterator = std::vector<ReactorFdState>::iterator;
using ConstFdIterator = std::vector<ReactorFdState>::const_iterator;

[[nodiscard]] OrderIterator FindOrder(ReactorRegistry &,
                                      std::uint64_t) noexcept;
[[nodiscard]] ConstOrderIterator FindOrder(const ReactorRegistry &,
                                           std::uint64_t) noexcept;
[[nodiscard]] FdIterator FindFd(ReactorRegistry &, ReactorHandle) noexcept;
[[nodiscard]] ConstFdIterator FindFd(const ReactorRegistry &,
                                     ReactorHandle) noexcept;
[[nodiscard]] bool FdMatches(ConstFdIterator, ConstFdIterator,
                             ReactorHandle) noexcept;
[[nodiscard]] bool FdMatches(FdIterator, FdIterator, ReactorHandle) noexcept;

[[nodiscard]] ReactorInterest Interest(const ReactorFdState &state) noexcept;
void AddInterest(ReactorFdState &state, ReactorInterest interest) noexcept;
void ResetSlot(ReactorRegistry &registry, std::uint32_t index) noexcept;
void ReleaseSlot(ReactorRegistry &registry, std::uint32_t index) noexcept;
[[nodiscard]] bool Unlink(ReactorRegistry &registry, ReactorFdState &fd,
                          std::uint32_t index) noexcept;

} // namespace rund::node::reactor_registry_detail
