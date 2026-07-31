#pragma once

#include "../lease.hpp"
#include "../many.hpp"

#include <rund/net/ready/many.hpp>

#include <span>
#include <vector>

namespace rund::node {

[[nodiscard]] ReactorManyGroup *
ReactorManyFindGroup(std::vector<ReactorManyGroup> &groups,
                     std::uint64_t group_id) noexcept;

[[nodiscard]] const ReactorManyGroup *
ReactorManyFindGroup(const std::vector<ReactorManyGroup> &groups,
                     std::uint64_t group_id) noexcept;

[[nodiscard]] std::span<ReactorManyRequest>
ReactorManyRequests(std::vector<ReactorManyRequest> &requests,
                    const ReactorManyGroup &group) noexcept;

[[nodiscard]] std::span<const ReactorManyRequest>
ReactorManyRequests(const std::vector<ReactorManyRequest> &requests,
                    const ReactorManyGroup &group) noexcept;

[[nodiscard]] ReasonCode
ReactorManyValidateRequests(std::span<const ReactorManyRequest> requests,
                            std::vector<std::uint32_t> &index_scratch,
                            std::uint64_t *comparisons) noexcept;

[[nodiscard]] bool
ReactorManyBuildRequests(std::span<const ::rund::net::ready::Request> requests,
                         std::span<const ::rund::net::SocketLease> leases,
                         std::vector<ReactorManyRequest> &out) noexcept;

[[nodiscard]] const ReactorManyRequest *
ReactorManyFindRequest(std::span<const ReactorManyRequest> requests,
                       std::uint64_t wait_id) noexcept;

void ReactorManyEraseGroup(std::vector<ReactorManyGroup> &groups,
                           std::vector<ReactorManyRequest> &requests,
                           std::uint64_t group_id) noexcept;

} // namespace rund::node
