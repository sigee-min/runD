#pragma once

#include "../../state/storage.hpp"

#include "../../../../replay/host/payload/hash.hpp"
#include "../../../../replay/input/plan.hpp"

#include <rund/replay/code.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::scheduler_replay_detail {

using PayloadCapture = ::rund::node::replay_detail::payload::Capture;
using InputBinding = ::rund::node::replay_detail::payload::InputBinding;
using InputSourceRange = ::rund::node::replay_detail::payload::InputSourceRange;
using MatchResult = ::rund::node::replay_detail::payload::MatchResult;
using ResolveResult = ::rund::node::replay_detail::payload::ResolveResult;
using PayloadBytes = ::rund::node::replay_detail::payload::Bytes;

[[nodiscard]] MatchResult poison_input(SchedulerState &,
                                       ::rund::replay::Code) noexcept;

[[nodiscard]] MatchResult store_replay_input(SchedulerState &,
                                             const InputBinding &,
                                             InputSourceRange, PayloadBytes,
                                             PayloadCapture) noexcept;

[[nodiscard]] bool same_identity(const InputBinding &,
                                 const InputBinding &) noexcept;

[[nodiscard]] bool to_size(std::uint64_t, std::size_t &) noexcept;

[[nodiscard]] bool can_account_input(const SchedulerState &,
                                     std::size_t) noexcept;

void account_input(SchedulerState &, std::size_t) noexcept;

[[nodiscard]] std::uint64_t
simulation_fingerprint(const SchedulerState &) noexcept;

} // namespace rund::node::scheduler_replay_detail
