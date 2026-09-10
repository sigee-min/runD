#include "local.hpp"

#include <kernel/program/compute/sort/model.hpp>

namespace rund::kernel {

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

} // namespace rund::kernel
