#include "local.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>

namespace rund::kernel {

[[nodiscard]] GraphSignature
GraphSignatureFor(const ReducePlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Reduce, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.element_count));
  if (plan.count_source != ComputeCountSource::Descriptor) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::LogicalCount, BufferRole::Read,
                 ComputeCountBytes(plan.count_source), 1u));
  }
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Output, BufferRole::Write,
                                    plan.element_bytes, 1u));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const ScatterReducePlan &plan) noexcept {
  GraphSignature out = graph_signature_detail::Begin(NodeKind::ScatterReduce,
                                                     plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Indices, BufferRole::Read,
                                    plan.index_bytes, plan.element_count));
  if (plan.count_source != ComputeCountSource::Descriptor) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::LogicalCount, BufferRole::Read,
                 ComputeCountBytes(plan.count_source), 1u));
  }
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Output, BufferRole::Write,
                                    plan.element_bytes, plan.output_count));
  return out;
}

} // namespace rund::kernel
