#include <kernel/program/compute/graph/signature.hpp>

#include <kernel/program/compute/compact/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/gather/model.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/histogram/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/metadata.hpp>
#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/scatter/model.hpp>
#include <kernel/program/compute/scatter/reduce/model.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>
#include <kernel/program/compute/solve/model.hpp>
#include <kernel/program/compute/sort/model.hpp>
#include <kernel/program/compute/spectrum/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/transform/model.hpp>

#include <cstddef>

namespace rund::kernel {

namespace graph_signature_detail {

[[nodiscard]] constexpr GraphValueType
Value(const GraphValueKind kind, const BufferRole role, const u64 element_bytes,
      const u64 count, const u64 rows = 0u, const u64 cols = 0u,
      const u64 batch_count = 0u) noexcept {
  return GraphValueType{
      .kind = kind,
      .role = role,
      .element_bytes = static_cast<u32>(element_bytes),
      .count = count,
      .rows = rows,
      .cols = cols,
      .batch_count = batch_count,
  };
}

[[nodiscard]] GraphSignature Reject(const NodeKind kind,
                                    const char *const reason) noexcept {
  return GraphSignature{.kind = kind, .reason = reason};
}

constexpr void Add(GraphSignature &signature,
                   const GraphValueType value) noexcept {
  if (signature.value_count >= kMaxGraphSignatureValues ||
      value.element_bytes > static_cast<u64>(~u32{0u})) {
    signature.ok = false;
    signature.reason = "compute_graph_signature_invalid";
    return;
  }
  signature.values[signature.value_count] = value;
  ++signature.value_count;
  if (value.role == BufferRole::Write) {
    ++signature.output_count;
  }
  if (value.kind == GraphValueKind::Status) {
    signature.status_count = value.count;
  }
}

[[nodiscard]] GraphSignature Begin(const NodeKind kind, const bool plan_ok,
                                   const char *const plan_reason) noexcept {
  return plan_ok ? GraphSignature{.kind = kind, .ok = true, .reason = "ok"}
                 : Reject(kind, plan_reason);
}

} // namespace graph_signature_detail

[[nodiscard]] GraphSignature
GraphSignatureFor(const ExecutionMetadata &metadata) noexcept {
  if (!metadata.ok ||
      metadata.binding_accesses.size() != metadata.binding_names.size() ||
      metadata.read_count > kMaxGraphSignatureValues ||
      metadata.write_count == 0u ||
      metadata.binding_accesses.size() > kMaxGraphSignatureValues ||
      metadata.input_element_bytes.size() !=
          static_cast<std::size_t>(metadata.read_count) ||
      metadata.output_element_bytes.size() !=
          static_cast<std::size_t>(metadata.write_count)) {
    return GraphSignature{.kind = NodeKind::Map, .reason = metadata.reason};
  }

  GraphSignature out{
      .kind = NodeKind::Map,
      .ok = true,
      .reason = "ok",
  };
  std::size_t read_index = 0u;
  std::size_t write_index = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const BufferRole role = GraphRoleFor(metadata.binding_accesses[index]);
    u64 element_bytes = 0u;
    GraphValueKind kind = GraphValueKind::Values;
    if (role == BufferRole::Read) {
      if (read_index >= metadata.input_element_bytes.size()) {
        return GraphSignature{.kind = NodeKind::Map,
                              .reason = "compute_graph_signature_invalid"};
      }
      element_bytes = metadata.input_element_bytes[read_index];
      ++read_index;
    } else if (role == BufferRole::Write) {
      if (write_index >= metadata.output_element_bytes.size()) {
        return GraphSignature{.kind = NodeKind::Map,
                              .reason = "compute_graph_signature_invalid"};
      }
      element_bytes = metadata.output_element_bytes[write_index];
      ++write_index;
      kind = GraphValueKind::Output;
    } else {
      return GraphSignature{.kind = NodeKind::Map,
                            .reason = "compute_graph_signature_invalid"};
    }
    if (element_bytes == 0u || element_bytes > static_cast<u64>(~u32{0u})) {
      return GraphSignature{.kind = NodeKind::Map,
                            .reason = "compute_graph_signature_invalid"};
    }
    out.values[out.value_count] = GraphValueType{
        .kind = kind,
        .role = role,
        .element_bytes = static_cast<u32>(element_bytes),
        .count = 0u,
    };
    ++out.value_count;
    if (role == BufferRole::Write) {
      ++out.output_count;
    }
  }
  return out;
}

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

[[nodiscard]] GraphSignature GraphSignatureFor(const SortPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Sort, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  const bool identity_values = plan.value == SortValue::IdentityU32;
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(GraphValueKind::Keys, BufferRole::Read,
                                         plan.key_bytes, plan.element_count));
  if (plan.count_source != ComputeCountSource::Descriptor) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::LogicalCount, BufferRole::Read,
                 ComputeCountBytes(plan.count_source), 1u));
  }
  if (!identity_values) {
    graph_signature_detail::Add(
        out,
        graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Read,
                                      plan.value_bytes, plan.element_count));
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Keys, BufferRole::Write,
                                       plan.key_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Values, BufferRole::Write,
                                    plan.value_bytes, plan.element_count));
  return out;
}

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

[[nodiscard]] GraphSignature
GraphSignatureFor(const ScatterReducePlan &plan) noexcept {
  GraphSignature out = graph_signature_detail::Begin(
      NodeKind::ScatterReduce, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(GraphValueKind::Values,
                                         BufferRole::Read,
                                         plan.element_bytes,
                                         plan.element_count));
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(GraphValueKind::Indices,
                                         BufferRole::Read, plan.index_bytes,
                                         plan.element_count));
  if (plan.count_source != ComputeCountSource::Descriptor) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::LogicalCount, BufferRole::Read,
                 ComputeCountBytes(plan.count_source), 1u));
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(GraphValueKind::Output,
                                         BufferRole::Write,
                                         plan.element_bytes,
                                         plan.output_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const StencilPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Stencil, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.element_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Output, BufferRole::Write,
                                    plan.element_bytes, plan.element_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const TransformPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Transform, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  if (plan.layout == TransformLayout::Split) {
    graph_signature_detail::Add(
        out,
        graph_signature_detail::Value(GraphValueKind::Real, BufferRole::Read,
                                      plan.element_bytes, plan.element_count));
    graph_signature_detail::Add(
        out,
        graph_signature_detail::Value(GraphValueKind::Imag, BufferRole::Read,
                                      plan.element_bytes, plan.element_count));
    graph_signature_detail::Add(
        out,
        graph_signature_detail::Value(GraphValueKind::Real, BufferRole::Write,
                                      plan.element_bytes, plan.element_count));
    graph_signature_detail::Add(
        out,
        graph_signature_detail::Value(GraphValueKind::Imag, BufferRole::Write,
                                      plan.element_bytes, plan.element_count));
  } else {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(GraphValueKind::Values,
                                           BufferRole::Read, plan.element_bytes,
                                           plan.element_count * 2u));
    graph_signature_detail::Add(out, graph_signature_detail::Value(
                                         GraphValueKind::Output,
                                         BufferRole::Write, plan.element_bytes,
                                         plan.element_count * 2u));
  }
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const MatrixPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Matrix, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Matrix, BufferRole::Read, plan.element_bytes,
               plan.left_count, plan.rows, plan.inner, plan.batch_count));
  if (plan.op != MatrixOp::Transpose) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::Matrix, BufferRole::Read, plan.element_bytes,
                 plan.right_count, plan.inner, plan.cols, plan.batch_count));
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Output, BufferRole::Write, plan.element_bytes,
               plan.output_count, plan.rows, plan.cols, plan.batch_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const FactorPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Factor, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Matrix, BufferRole::Read,
                                       plan.element_bytes, plan.input_count,
                                       plan.rows, plan.cols, plan.batch_count));
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Factor, BufferRole::Write, plan.element_bytes,
               plan.factor_count, plan.rows, plan.cols, plan.batch_count));
  if (plan.aux_count != 0u) {
    graph_signature_detail::Add(out, graph_signature_detail::Value(
                                         GraphValueKind::Aux, BufferRole::Write,
                                         sizeof(u32), plan.aux_count));
  }
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Status, BufferRole::Write,
                                    sizeof(u32), plan.status_count));
  return out;
}

[[nodiscard]] GraphSignature GraphSignatureFor(const SolvePlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Solve, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               plan.input == SolveInput::Factor ? GraphValueKind::Factor
                                                : GraphValueKind::Matrix,
               BufferRole::Read, plan.element_bytes,
               plan.input == SolveInput::Factor ? plan.factor_count
                                                : plan.matrix_count,
               plan.rows, plan.rows, plan.batch_count));
  if (plan.input == SolveInput::Factor && plan.aux_count != 0u) {
    graph_signature_detail::Add(out, graph_signature_detail::Value(
                                         GraphValueKind::Aux, BufferRole::Read,
                                         sizeof(u32), plan.aux_count));
  }
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Rhs, BufferRole::Read, plan.element_bytes,
               plan.rhs_count, plan.rows, plan.rhs_cols, plan.batch_count));
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Output, BufferRole::Write, plan.element_bytes,
               plan.output_count, plan.rows, plan.rhs_cols, plan.batch_count));
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Status, BufferRole::Write,
                                    sizeof(u32), plan.status_count));
  return out;
}

[[nodiscard]] GraphSignature
GraphSignatureFor(const SpectrumPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Spectrum, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Matrix, BufferRole::Read,
                                       plan.element_bytes, plan.input_count,
                                       plan.rows, plan.cols, plan.batch_count));
  graph_signature_detail::Add(
      out, graph_signature_detail::Value(
               GraphValueKind::Values, BufferRole::Write, plan.element_bytes,
               plan.value_count, plan.rows, 1u, plan.batch_count));
  if (plan.vector_count != 0u) {
    graph_signature_detail::Add(
        out, graph_signature_detail::Value(
                 GraphValueKind::Vectors, BufferRole::Write, plan.element_bytes,
                 plan.vector_count, plan.rows, plan.cols, plan.batch_count));
  }
  graph_signature_detail::Add(
      out,
      graph_signature_detail::Value(GraphValueKind::Status, BufferRole::Write,
                                    sizeof(u32), plan.status_count));
  return out;
}

} // namespace rund::kernel
