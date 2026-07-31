#pragma once

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/fusion/result.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

struct AdmittedFusedMapChainIR {
  ComputeFusedMapChainIR value{};
  ComputeInputAdmission input{};
  u32 source_parse_count = 0u;
};

[[nodiscard]] AdmittedFusedMapChainIR
BuildAdmittedFusedComputeMapChainIR(const ComputeIR *chain, u64 chain_count,
                                    const Graph &graph,
                                    const FusionPolicy &policy,
                                    ComputeApi api);

[[nodiscard]] AdmittedFusedMapChainIR
BuildAdmittedFusedComputeMapChainIR(const ComputeIR *const *chain,
                                    u64 chain_count, const Graph &graph,
                                    const FusionPolicy &policy,
                                    ComputeApi api);

[[nodiscard]] AdmittedFusedMapChainIR BuildAdmittedFusedComputeMapChainIR(
    const ComputeIR *const *chain, ComputeInputAdmission *const *inputs,
    u64 chain_count, const Graph &graph, const FusionPolicy &policy,
    ComputeApi api);

} // namespace compute_lowering_detail

[[nodiscard]] ComputeFusedMapChainIR
BuildFusedComputeMapChainIR(const ComputeIR *chain, u64 chain_count,
                            const Graph &graph, const FusionPolicy &policy,
                            ComputeApi api);

} // namespace rund::kernel
