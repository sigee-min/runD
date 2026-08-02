#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "source/base.hpp"
#include "source/count.hpp"
#include "source/scatter.hpp"
#include "source/status.hpp"
#include "../../kernel/backend/source_recipe.hpp"
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {
template <typename Sink> [[nodiscard]] bool EmitMetalCompactSource(Sink &sink) {
  return sink.append(MetalCompactBaseSource()) &&
         sink.append(MetalCompactCountSource()) &&
         sink.append(MetalCompactScatterSource()) &&
         sink.append(MetalCompactStatusSource());
}
} // namespace

[[nodiscard]] std::string MetalCompactSource() {
  return backend_source_recipe::materialize(
      [](auto &sink) { return EmitMetalCompactSource(sink); });
}

bool MetalCompactSourceUpperBytes(std::uint64_t &upper) noexcept {
  return backend_source_recipe::bytes(
      [](auto &sink) noexcept { return EmitMetalCompactSource(sink); }, upper);
}
#endif

} // namespace rund::node::accel::detail
