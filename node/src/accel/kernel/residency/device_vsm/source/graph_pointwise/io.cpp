#include "internal.hpp"

#include <kernel/program/compute/lowering/names.hpp>

namespace rund::node::accel::detail::device_vsm_graph_pointwise {

rund::kernel::compute_lowering_detail::ParsedIR
build_io_ir(const std::size_t external_input_count,
            const rund::kernel::ComputeScalar scalar,
            const rund::kernel::ComputeDomain domain,
            const std::uint32_t element_bytes) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (!((scalar == rund::kernel::ComputeScalar::Lane32 &&
         domain == rund::kernel::ComputeDomain::U32 && element_bytes == 4u) ||
        (scalar == rund::kernel::ComputeScalar::Lane64 &&
         domain == rund::kernel::ComputeDomain::U64 && element_bytes == 8u))) {
    return {};
  }
  const std::uint8_t scalar_mode = lowering::DomainModeFor(scalar, domain);
  lowering::ParsedIR result{
      .name = "device-vsm-graph-pointwise-io",
      .scalar_mode = scalar_mode,
      .ok = true,
      .reason = "ok",
  };
  result.bindings.reserve(external_input_count + 1u);
  for (std::size_t input = 0u; input < external_input_count; ++input) {
    result.bindings.push_back(lowering::ParsedBinding{
        .kind = lowering::kReadBindingKind,
        .numeric_mode = result.scalar_mode,
        .name = "rund_graph_input_" + std::to_string(input),
        .element_bytes = element_bytes,
    });
  }
  result.bindings.push_back(lowering::ParsedBinding{
      .kind = lowering::kWriteBindingKind,
      .numeric_mode = result.scalar_mode,
      .name = "rund_graph_output",
      .element_bytes = element_bytes,
  });
  return result;
}

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise
