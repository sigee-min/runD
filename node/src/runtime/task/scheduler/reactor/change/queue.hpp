#pragma once

#include "../backend.hpp"

namespace rund::node {

[[nodiscard]] ReactorApplyResult
ReactorChangeQueueApply(ReactorRuntime &reactor,
                        ::rund::detail::task::StatStorage &stats) noexcept;

[[nodiscard]] bool
ReactorChangeQueueAcknowledgeInvalid(ReactorRuntime &reactor,
                                     ReactorInvalidChangeToken token) noexcept;

} // namespace rund::node
