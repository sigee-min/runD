#pragma once

#include <rund/host/event.hpp>
#include <rund/net/vectored.hpp>

#include "../../../runtime/platform/net.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::net {

struct NetEventRequest {
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::uint64_t socket_id = 0u;
  node::NativeCallResult native{};
  std::uint64_t requested_bytes = 0u;
  ::rund::StableHash name_hash{};
  ::rund::StableHash payload_hash{};
};

[[nodiscard]] bool RecordNetEvent(NetEventRequest request) noexcept;
[[nodiscard]] bool
RecordNetIngressEvent(NetEventRequest request,
                      std::span<const std::byte> bytes) noexcept;
[[nodiscard]] bool
RecordNetIngressEvent(NetEventRequest request,
                      std::span<const batch::Buffer> slices) noexcept;

} // namespace rund::net
