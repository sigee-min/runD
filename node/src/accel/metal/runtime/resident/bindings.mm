#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/find.hpp"
#include "../../resident/access.hpp"
#include "../local.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
bool UploadMetalBufferUncounted(const MetalRuntimeBuffer &buffer,
                                const void *const data,
                                const rund::kernel::u64 bytes) {
  if (bytes == 0u) {
    return true;
  }
  if (buffer.buffer == nullptr || data == nullptr || bytes > buffer.bytes) {
    return false;
  }
  void *const contents = MetalBufferContents(buffer);
  if (contents == nullptr) {
    return false;
  }
  std::memcpy(contents, data, static_cast<std::size_t>(bytes));
  return true;
}

bool PrepareResidentBindings(MetalAdapter &adapter,
                             const rund::kernel::ComputePlan &plan,
                             const rund::kernel::BindingSet &bindings,
                             MetalResidentBindings &out) {
  if ((plan.input_buffer_count != 0u &&
       (!bindings.resident_inputs.has_refs() ||
        !bindings.resident_inputs.has_handles())) ||
      bindings.resident_inputs.count != plan.input_buffer_count ||
      !bindings.resident_outputs.has_refs() ||
      !bindings.resident_outputs.has_handles() ||
      bindings.resident_outputs.count != plan.output_buffer_count ||
      plan.output_buffer_count == 0u) {
    return false;
  }
  out = MetalResidentBindings{};
  out.bindings = &bindings;
  if (plan.input_buffer_count > kInlineMetalBufferCount) {
    std::size_t overflow_count = 0u;
    if (!ToSize(plan.input_buffer_count - kInlineMetalBufferCount,
                overflow_count)) {
      return false;
    }
    out.overflow_inputs.resize(overflow_count);
    if (out.overflow_inputs.capacity() != overflow_count) {
      return false;
    }
  }
  if (plan.output_buffer_count > kInlineMetalBufferCount) {
    std::size_t overflow_count = 0u;
    if (!ToSize(plan.output_buffer_count - kInlineMetalBufferCount,
                overflow_count)) {
      return false;
    }
    out.overflow_outputs.resize(overflow_count);
    if (out.overflow_outputs.capacity() != overflow_count) {
      return false;
    }
  }
  MetalResidentState &resident = MetalResidents(adapter);
  std::lock_guard lock{resident.mutex};
  for (rund::kernel::u64 index = 0u; index < plan.output_buffer_count;
       ++index) {
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_outputs.ref(index);
    const std::shared_ptr<void> *const handle =
        bindings.resident_outputs.handle(index);
    if (ref == nullptr || handle == nullptr || *handle == nullptr ||
        ref->stride_bytes < ref->element_bytes) {
      return false;
    }
    MetalResidentBufferResult &slot =
        index < kInlineMetalBufferCount
            ? out.outputs[static_cast<std::size_t>(index)]
            : out.overflow_outputs[static_cast<std::size_t>(
                  index - kInlineMetalBufferCount)];
    slot = ResolveMetalResidentBuffer(resident, *ref, *handle,
                                      "accel_metal_resident_id_unavailable",
                                      true);
    if (!slot.check.ok || slot.device_buffer == nullptr) {
      return false;
    }
  }
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_inputs.ref(index);
    const std::shared_ptr<void> *const handle =
        bindings.resident_inputs.handle(index);
    if (ref == nullptr || handle == nullptr || *handle == nullptr ||
        ref->stride_bytes < ref->element_bytes) {
      return false;
    }
    MetalResidentBufferResult &slot =
        index < kInlineMetalBufferCount
            ? out.inputs[static_cast<std::size_t>(index)]
            : out.overflow_inputs[static_cast<std::size_t>(
                  index - kInlineMetalBufferCount)];
    slot = ResolveMetalResidentBuffer(resident, *ref, *handle,
                                      "accel_metal_resident_id_unavailable",
                                      true);
    if (!slot.check.ok || slot.device_buffer == nullptr) {
      return false;
    }
  }
  return true;
}
#endif

} // namespace rund::node::accel::detail
