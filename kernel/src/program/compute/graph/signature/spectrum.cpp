#include "local.hpp"

#include <kernel/program/compute/spectrum/model.hpp>

namespace rund::kernel {

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
