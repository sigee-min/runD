#include "local.hpp"

#include <kernel/program/compute/transform/model.hpp>

namespace rund::kernel {

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

} // namespace rund::kernel
