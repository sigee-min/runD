#pragma once

#include "../model.hpp"

#include "../../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool EmitMetalSegmentedReduceReduceSource(
    backend_source_recipe::CountSink &sink, rund::kernel::ReduceOp op,
    rund::kernel::ComputeDomain domain) noexcept;
[[nodiscard]] bool EmitMetalSegmentedReduceReduceSource(
    backend_source_recipe::StringSink &sink, rund::kernel::ReduceOp op,
    rund::kernel::ComputeDomain domain);

#endif

} // namespace rund::node::accel::detail
