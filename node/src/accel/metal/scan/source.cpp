#include "source.hpp"

#include "source/base.hpp"
#include "source/32/program.hpp"
#include "source/64/program.hpp"
#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

namespace {
template <typename Sink> [[nodiscard]] bool EmitMetalScanSource(Sink &sink) {
  return sink.append(MetalScanBaseSource()) &&
         sink.append(MetalScanBlockU32Source()) &&
         sink.append(MetalScanBlockFlagU32Source()) &&
         sink.append(MetalScanPrefixU32Source()) &&
         sink.append(MetalScanOffsetU32Source()) &&
         sink.append(MetalScanBlockU64Source()) &&
         sink.append(MetalScanPrefixU64Source()) &&
         sink.append(MetalScanOffsetU64Source());
}
} // namespace

std::string MetalScanSource() {
  return backend_source_recipe::materialize(
      [](auto &sink) { return EmitMetalScanSource(sink); });
}

bool MetalScanSourceUpperBytes(std::uint64_t &upper) noexcept {
  return backend_source_recipe::bytes(
      [](auto &sink) noexcept { return EmitMetalScanSource(sink); }, upper);
}

} // namespace rund::node::accel::detail
