#include "local.hpp"

#include <kernel/program/compute/stencil/model.hpp>
#include <kernel/program/compute/window/model.hpp>

namespace rund::kernel {

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
GraphSignatureFor(const WindowPlan &plan) noexcept {
  GraphSignature out =
      graph_signature_detail::Begin(NodeKind::Window, plan.ok, plan.reason);
  if (!out.ok) {
    return out;
  }
  graph_signature_detail::Add(out, graph_signature_detail::Value(
                                       GraphValueKind::Values, BufferRole::Read,
                                       plan.element_bytes, plan.input_count));
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
