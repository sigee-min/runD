#pragma once

#include "model.hpp"

#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/runtime.hpp>

namespace rund::node::accel::detail::prepared {

void Accumulate(EvidenceCounts &counts, const RunState &state) noexcept;
void Accumulate(EvidenceCounts &counts, const RunState &state,
                std::uint64_t occurrences) noexcept;

[[nodiscard]] rund::AccelEvidence
BatchEvidence(const rund::AccelContext &context, rund::RuntimeStats stats,
              const EvidenceCounts &counts, rund::AccelCheck check) noexcept;

[[nodiscard]] rund::AccelEvidence RunEvidence(const rund::AccelContext &context,
                                              const RunState &state,
                                              const KernelResult &run);

[[nodiscard]] PreparedPipelineEvidence
PipelineEvidence(const rund::AccelContext &context,
                 const PipelineState &pipeline,
                 const KernelResult &backend) noexcept;

} // namespace rund::node::accel::detail::prepared
