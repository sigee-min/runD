#pragma once

#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include "../../context/shared.hpp"
#include "../admission.hpp"
#include "../collective.hpp"
#include <kernel/program/compute/graph/signature.hpp>
#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] bool SameBindingName(const char *lhs,
                                   const std::string &rhs) noexcept;
[[nodiscard]] bool
BindingOrderOk(const rund::AccelGraphNode &node,
               const rund::kernel::ExecutionMetadata &metadata) noexcept;
[[nodiscard]] bool PrimitivePayloadOnly(const rund::AccelGraphNode &node,
                                        rund::kernel::NodeKind active) noexcept;

struct AdmissionSignature {
  rund::kernel::GraphSignature signature{};
  bool ok = false;
};

[[nodiscard]] inline AdmissionSignature
AdmitSignature(const rund::AccelGraphNode &node,
               const rund::kernel::GraphSignature &signature) noexcept {
  return AdmissionSignature{
      .signature = signature,
      .ok = NodeSignatureOk(node, signature),
  };
}

template <typename Plan>
[[nodiscard]] AdmissionSignature
AdmitSignature(const rund::AccelGraphNode &node, const Plan &plan) noexcept {
  const rund::kernel::GraphSignature signature =
      rund::kernel::GraphSignatureFor(plan);
  return AdmitSignature(node, signature);
}

[[nodiscard]] const char *AdmitMapNode(const rund::AccelGraphNode &node,
                                       rund::kernel::ComputeApi api,
                                       GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitScanNode(const rund::AccelGraphNode &node,
                                        GraphCompileNode &compile_data);
[[nodiscard]] const char *
AdmitSegmentedScanNode(const rund::AccelGraphNode &node,
                       GraphCompileNode &compile_data);
[[nodiscard]] const char *
AdmitSegmentedReduceNode(const rund::AccelGraphNode &node,
                         GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitSortNode(const rund::AccelGraphNode &node,
                                        GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitCompactNode(const rund::AccelGraphNode &node,
                                           GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitGatherNode(const rund::AccelGraphNode &node,
                                          GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitHistogramNode(const rund::AccelGraphNode &node,
                                             GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitPartitionNode(const rund::AccelGraphNode &node,
                                             GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitReduceNode(const rund::AccelGraphNode &node,
                                          GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitScatterNode(const rund::AccelGraphNode &node,
                                           GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitScatterReduceNode(
    const rund::AccelGraphNode &node, GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitStencilNode(const rund::AccelGraphNode &node,
                                           GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitTransformNode(const rund::AccelGraphNode &node,
                                             GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitMatrixNode(const rund::AccelGraphNode &node,
                                          GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitFactorNode(const rund::AccelGraphNode &node,
                                          GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitSolveNode(const rund::AccelGraphNode &node,
                                         GraphCompileNode &compile_data);
[[nodiscard]] const char *AdmitSpectrumNode(const rund::AccelGraphNode &node,
                                            GraphCompileNode &compile_data);

} // namespace rund::node::accel::detail
