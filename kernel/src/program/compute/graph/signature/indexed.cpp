#include "local.hpp"

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>

namespace rund::kernel {

[[nodiscard]] GraphSignature
GraphSignatureFor(const CompactPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Compact, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Flags, BufferRole::Read,
                                       plan.flag_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Write,
                                    plan.output_bytes, plan.output_capacity));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const GatherPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Gather, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.source_count));
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
                                    plan.element_bytes, plan.element_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const HistogramPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Histogram, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(GraphValueKind::Bins, BufferRole::Read,
                                         plan.index_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Counts, BufferRole::Write,
                                    plan.count_bytes, plan.bin_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const PartitionPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Partition, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Flags, BufferRole::Read,
                                       plan.flag_bytes, plan.element_count));
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.value_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Output, BufferRole::Write,
                                    plan.value_bytes, plan.element_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const ScatterPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Scatter, plan.ok, plan.reason);
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
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Output, BufferRole::Write,
                                    plan.element_bytes, plan.output_count));
  return out;
}

} // namespace rund::kernel
