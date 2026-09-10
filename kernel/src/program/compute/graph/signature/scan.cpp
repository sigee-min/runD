#include "local.hpp"

#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>

namespace rund::kernel {

[[nodiscard]] GraphSignature GraphSignatureFor(const ScanPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Scan, plan.ok, plan.reason);
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
      graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Write,
                                    plan.element_bytes, plan.element_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const SegmentedScanPlan &plan) noexcept {
  GraphSignature out = graph_signature_detail::Begin(NodeKind::SegmentedScan,
                                                     plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.element_count));
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Heads, BufferRole::Read,
                                       plan.head_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Write,
                                    plan.element_bytes, plan.element_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const SegmentedReducePlan &plan) noexcept {
  GraphSignature out = graph_signature_detail::Begin(NodeKind::SegmentedReduce,
                                                     plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.element_count));
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Heads, BufferRole::Read,
                                       plan.head_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Write,
                                    plan.element_bytes, plan.element_count));
  return out;
}

} // namespace rund::kernel
