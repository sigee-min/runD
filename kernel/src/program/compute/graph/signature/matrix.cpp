#include "local.hpp"

#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>
#include <kernel/program/compute/solve/model.hpp>

namespace rund::kernel {

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

} // namespace rund::kernel
