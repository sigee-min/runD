#pragma once

#include "../../state/storage.hpp"

#include <rund/task/channel/access.hpp>

namespace rund::node::channel_access_detail {

[[nodiscard]] ::rund::detail::task::ChannelDecision
MissingNodeRuntimeResult() noexcept;

} // namespace rund::node::channel_access_detail
