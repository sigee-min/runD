#include "../../scan/metal.hpp"
#include "../adapter.hpp"

namespace rund::node::accel::detail {

rund::kernel::u32 MetalScanStatusFlags(const MetalRuntimeBuffer &status) {
  const auto *const values =
      static_cast<const rund::kernel::u32 *>(MetalBufferContents(status));
  if (values == nullptr) {
    return ~rund::kernel::u32{0u};
  }
  return values[0];
}

bool MetalScanStatusOk(const MetalRuntimeBuffer &status) {
  return MetalScanStatusFlags(status) == 0u;
}

} // namespace rund::node::accel::detail
