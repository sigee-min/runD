#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

void classify_device_vsm_backing_evidence(
    const DeviceVsmProductOwner &owner,
    DeviceVsmProductEvidence &evidence) noexcept {
  evidence.public_resident_input_count = 0u;
  evidence.whole_run_staged_input_count = 0u;
  for (std::size_t index = 0u; index < owner.input_count; ++index) {
    if (owner.resident_inputs[index] != nullptr) {
      ++evidence.public_resident_input_count;
    } else {
      ++evidence.whole_run_staged_input_count;
    }
  }
  evidence.public_resident_output = owner.resident_output != nullptr;
  evidence.whole_run_staged_output = owner.resident_output == nullptr;
  evidence.bounded_external_page_service = false;
}

} // namespace rund::compute::detail::device_vsm_product_detail
