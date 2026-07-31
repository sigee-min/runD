#include "model.hpp"

#include "../../fixed/format.hpp"
#include "../../type.hpp"
#include "../local.hpp"

#include <kernel/program/compute/graph/identity.hpp>
#include <kernel/program/compute/graph/validation.hpp>
#include <rund/counter.hpp>

#include <new>
#include <utility>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {
namespace {

[[nodiscard]] Description failure(const Reason reason) {
  return Description{.status = Status::fail(reason)};
}

[[nodiscard]] kernel::graph_detail::GraphHash
identify(graph::Info &info, kernel::graph_detail::GraphHash fingerprint) {
  fingerprint = kernel::graph_detail::Mix(fingerprint, 0x72756e4447525032ull);
  // Cached Program observations retain the authored admission count, so that
  // count must share the same canonical identity owner.
  fingerprint = kernel::graph_detail::Mix(fingerprint, info.authored_nodes);
  for (const graph::Resource &resource : info.resources) {
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.id);
    fingerprint = kernel::graph_detail::Mix(
        fingerprint, static_cast<std::uint64_t>(resource.type));
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.integer_bits);
    fingerprint =
        kernel::graph_detail::Mix(fingerprint, resource.fraction_bits);
    fingerprint = kernel::graph_detail::Mix(
        fingerprint, static_cast<std::uint64_t>(resource.rounding));
    fingerprint = kernel::graph_detail::Mix(
        fingerprint, static_cast<std::uint64_t>(resource.overflow));
    fingerprint = kernel::graph_detail::Mix(
        fingerprint, static_cast<std::uint64_t>(resource.approximation));
    fingerprint = kernel::graph_detail::Mix(
        fingerprint, static_cast<std::uint64_t>(resource.visibility));
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.elements);
    fingerprint =
        kernel::graph_detail::Mix(fingerprint, resource.element_bytes);
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.active);
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.parent);
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.source);
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.alias_group);
    fingerprint =
        kernel::graph_detail::Mix(fingerprint, resource.alias_offset_bytes);
    fingerprint = kernel::graph_detail::Mix(fingerprint, resource.reset_node);
  }
  for (const graph::Node &node : info.nodes) {
    for (const graph::Access &access : node.accesses) {
      if (access.mode == resource::AccessMode::Read) {
        info.read_bytes = ::rund::detail::counter::SaturatingAdd(
            info.read_bytes, access.element_bytes * access.element_count);
      }
      fingerprint = kernel::graph_detail::Mix(fingerprint, node.index);
      fingerprint = kernel::graph_detail::Mix(fingerprint, access.resource);
      fingerprint = kernel::graph_detail::Mix(
          fingerprint, static_cast<std::uint64_t>(access.mode));
      fingerprint = kernel::graph_detail::Mix(fingerprint, access.offset_bytes);
      fingerprint =
          kernel::graph_detail::Mix(fingerprint, access.element_bytes);
      fingerprint =
          kernel::graph_detail::Mix(fingerprint, access.element_count);
      fingerprint = kernel::graph_detail::Mix(fingerprint, access.stride_bytes);
    }
  }
  for (const std::uint32_t input : info.inputs) {
    fingerprint = kernel::graph_detail::Mix(fingerprint, input);
  }
  for (const std::uint32_t output : info.outputs) {
    fingerprint = kernel::graph_detail::Mix(fingerprint, output);
  }
  return fingerprint;
}

} // namespace

Description finish(Draft draft, const Type root, const FixedFormat root_format,
                   const std::span<const std::uint32_t> identity_outputs,
                   const std::uint64_t page_bytes) {
  graph::Info &info = draft.description.info;
  const Status planned =
      resource_detail::plan_memory(info, draft.memory, page_bytes);
  if (!planned) {
    return failure(planned.reason());
  }
  const Status hazards = build_hazards(info);
  if (!hazards) {
    return failure(hazards.reason());
  }

  std::vector<kernel::u64> canonical_outputs;
  try {
    canonical_outputs.assign(identity_outputs.begin(), identity_outputs.end());
  } catch (const std::bad_alloc &) {
    return failure(Reason::GraphCapacity);
  }
  const kernel::Graph canonical{
      .nodes = draft.nodes.data(),
      .node_count = draft.nodes.size(),
      .outputs = canonical_outputs.data(),
      .output_count = canonical_outputs.size(),
      .scalar = type_scalar(root),
      .domain = type_domain(root),
      .fixed_format = kernel_format(root_format),
  };
  const kernel::GraphCheck check =
      draft.zero_work ? kernel::ValidateGraphIdentity(canonical)
                      : kernel::ValidateGraph(canonical);
  if (!check.ok) {
    return failure(project_reason(check.reason, Reason::GraphInvalid));
  }

  const kernel::graph_detail::GraphHash fingerprint =
      identify(info, kernel::graph_detail::GraphHash{
                         .hi = check.graph_id_hi,
                         .lo = check.graph_id_lo,
                     });
  info.fingerprint =
      graph::Fingerprint{.hi = fingerprint.hi, .lo = fingerprint.lo};
  if (!info.fingerprint) {
    info.fingerprint.lo = 1u;
  }
  return std::move(draft.description);
}

} // namespace rund::compute::detail::graph_detail::describe_detail
