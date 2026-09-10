#pragma once

namespace rund { struct AccelEvidence; }
namespace rund::node::accel::detail {
struct PreparedPipelineEvidence;
using PreparedKernelCompletion = void (*)(void *, const rund::AccelEvidence &) noexcept;
using PreparedPipelineCompletion = void (*)(void *, PreparedPipelineEvidence &&) noexcept;
using PreparedBatchStart = void (*)(void *) noexcept;
} // namespace rund::node::accel::detail
