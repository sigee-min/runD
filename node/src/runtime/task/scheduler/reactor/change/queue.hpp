#pragma once

#include "../backend.hpp"

namespace rund::node {

[[nodiscard]] ReactorApplyResult
ReactorChangeQueueApply(ReactorRuntime &reactor,
                        ::rund::detail::task::StatStorage &stats) noexcept;

} // namespace rund::node
