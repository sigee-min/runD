#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "source/base.hpp"
#include "source/count.hpp"
#include "source/scatter.hpp"
#include "source/status.hpp"
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] std::string MetalCompactSource() {
  std::string source;
  source += MetalCompactBaseSource();
  source += MetalCompactCountSource();
  source += MetalCompactScatterSource();
  source += MetalCompactStatusSource();
  return source;
}
#endif

} // namespace rund::node::accel::detail
