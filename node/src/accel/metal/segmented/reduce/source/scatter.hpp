#pragma once

#include "../../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] bool EmitMetalSegmentedReduceScatterSource(
    backend_source_recipe::CountSink &sink) noexcept;
[[nodiscard]] bool EmitMetalSegmentedReduceScatterSource(
    backend_source_recipe::StringSink &sink);

#endif

} // namespace rund::node::accel::detail
