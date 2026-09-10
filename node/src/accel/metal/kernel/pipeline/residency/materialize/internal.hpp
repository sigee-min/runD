#pragma once

#include "../local.hpp"

#include <cstdint>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace rund::node::accel::detail {

struct MetalResidencyCandidate final {
  MetalSequence *sequence{};
  MetalResidencySubmission submission{};
  MetalResidencySlidingGate sliding{};
  MetalResidencyWindow window{};
  std::uint64_t retained_bytes{};

  ~MetalResidencyCandidate();
};

} // namespace rund::node::accel::detail

#endif
