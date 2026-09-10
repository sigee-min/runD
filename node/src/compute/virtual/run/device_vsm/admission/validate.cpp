#include "validate.hpp"

#include "../../../../type.hpp"

namespace rund::compute::detail::device_vsm_product_detail::admission_detail {
namespace {

[[nodiscard]] constexpr ::rund::AccelCheck pass() noexcept {
  return {true, "ok"};
}

[[nodiscard]] constexpr ::rund::AccelCheck
fail(const char *const reason) noexcept {
  return {false, reason};
}

} // namespace

::rund::AccelCheck
check_bindings(const VirtualRunProjection &run, const std::uint64_t page_count,
               const std::size_t element_bytes, const bool graph_pointwise,
               const bool graph_map_reduce, const bool scan, const bool reduce,
               const std::shared_ptr<PipelineState> &selected_primary,
               const std::shared_ptr<PipelineState> &second,
               const bool authority) noexcept {
  if (page_count < 2u ||
      page_count > std::numeric_limits<std::uint32_t>::max()) {
    return fail("compute_bounded_count_invalid");
  }
  if (run.active.input_bytes == 0u) {
    return fail("compute_binding_input_bytes_mismatch");
  }
  if (!(graph_pointwise || graph_map_reduce || reduce) &&
      run.active.input_bytes != run.active.output_bytes) {
    return fail("compute_binding_output_bytes_mismatch");
  }
  if (run.input_payload_bytes == 0u) {
    return fail("compute_binding_input_bytes_mismatch");
  }
  if (!(graph_pointwise || graph_map_reduce || scan || reduce) &&
      run.input_payload_bytes != run.output_payload_bytes) {
    return fail("compute_binding_output_bytes_mismatch");
  }
  if (element_bytes == 0u || element_bytes != type_bytes(run.output_type)) {
    return fail("compute_binding_type_mismatch");
  }
  if (run.active.input_bytes % element_bytes != 0u ||
      run.input_payload_bytes % element_bytes != 0u) {
    return fail("compute_binding_input_bytes_mismatch");
  }
  if (selected_primary == nullptr || second == nullptr) {
    return fail("compute_pipeline_invalid");
  }
  if (selected_primary == second) {
    return fail("compute_binding_duplicate");
  }
  if (selected_primary->transactional || second->transactional) {
    return fail("compute_primitive_route_invalid");
  }
  if (selected_primary->device == nullptr ||
      selected_primary->device != second->device) {
    return fail("compute_binding_device_mismatch");
  }
  if (!authority) {
    return fail("compute_pipeline_invalid");
  }
  return pass();
}

} // namespace
  // rund::compute::detail::device_vsm_product_detail::admission_detail
