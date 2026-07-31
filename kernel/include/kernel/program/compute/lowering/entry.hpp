#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class ComputeApi : u8;
struct ComputeIR;
struct LoweringArtifact;

[[nodiscard]] LoweringArtifact LowerComputeIR(const ComputeIR &ir,
                                               ComputeApi api);

} // namespace rund::kernel
