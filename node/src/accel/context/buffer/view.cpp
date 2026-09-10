#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include "../../backend/usage.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

rund::AccelBuffer ProjectAccelBufferView(const rund::AccelContext &context,
                                         const rund::AccelBuffer &buffer,
                                         const rund::AccelBufferDesc desc) {
  const TransferAdmission admission = AdmitAccelBufferTransfer(context, buffer);
  if (!admission.check.ok) {
    return RejectBuffer(desc, buffer.buffer, admission.check.reason);
  }

  const rund::AccelCheck desc_check = CheckDesc(desc);
  if (!desc_check.ok) {
    return RejectBuffer(desc, buffer.buffer, desc_check.reason);
  }

  // A typed view may change width/count together, but it must cover exactly the
  // already admitted logical extent. The backend allocation may be larger;
  // exposing a partial or extended semantic view would create a new range
  // authority outside the source capability.
  const std::uint64_t view_extent = desc.scalar_width_bytes * desc.count;
  if (view_extent != admission.byte_extent ||
      !UsageCompatible(desc.usage, buffer.buffer.usage)) {
    return RejectBuffer(desc, buffer.buffer, "accel_context_buffer_invalid");
  }

  rund::Buffer backend = buffer.buffer;
  backend.owner = admission.pick;
  backend.handle = admission.route.handle;
  return OpenAccelBuffer(context, backend, desc);
}

} // namespace rund::node::accel::detail
