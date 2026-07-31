#pragma once

#include "../../../../../../host/net/interest.hpp"
#include "../../../../../../host/net/operation.hpp"
#include "../../../state/storage.hpp"
#include "../../cleanup/request.hpp"
#include "../../many/store.hpp"
#include "store.hpp"
#include "../../stats.hpp"

namespace rund::node {

[[nodiscard]] ::rund::net::ready::Status
ReadySetStatus(ReasonCode code, ::rund::net::ready::Set set = {}) noexcept;

[[nodiscard]] ::rund::net::ready::many::Result
ReadySetWaitStatus(ReasonCode code, std::uint32_t events = 0u,
                   bool budget_exhausted = false) noexcept;

[[nodiscard]] bool
ReadySetMemberIsCurrent(const ReactorReadySetMember &member) noexcept;

} // namespace rund::node
