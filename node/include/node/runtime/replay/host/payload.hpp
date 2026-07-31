#pragma once

#include <node/runtime/replay/host/evidence.hpp>
#include <rund/replay/code.hpp>

#include <span>
#include <string_view>
#include <vector>

namespace rund::node {

namespace replay_detail {

[[nodiscard]] bool IsPayloadKind(::rund::host::EventKind kind) noexcept;

[[nodiscard]] bool
EventRequiresPayload(const ::rund::host::Event &event) noexcept;

[[nodiscard]] ::rund::replay::Code BindPayloads(
    const std::vector<::rund::host::Event> &events,
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept;

[[nodiscard]] bool EventsRequirePayload(
    const std::vector<::rund::host::Event> &events) noexcept;

// Validates archive structure plus resident canonical hashes directly. Spill
// archives validate their sealed metadata here; the storage loader owns disk
// bytes and segment-record validation.
[[nodiscard]] bool
ValidArchive(const ::rund::node::replay_detail::payload::Archive &archive);

} // namespace replay_detail

} // namespace rund::node
