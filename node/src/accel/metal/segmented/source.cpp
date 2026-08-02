#include "local.hpp"
#include "source/block.hpp"
#include "source/offset.hpp"
#include "source/prefix.hpp"
#include "source/prelude.hpp"
#include "../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

namespace {
template <typename Sink>
[[nodiscard]] bool EmitMetalSegmentedScanSource(Sink &sink) {
  return sink.append(MetalSegmentedPrelude()) &&
         sink.append(MetalSegmentedBlockSource()) &&
         sink.append(MetalSegmentedPrefixSource()) &&
         sink.append(MetalSegmentedOffsetSource());
}
} // namespace

std::string MetalSegmentedScanSource() {
  return backend_source_recipe::materialize(
      [](auto &sink) { return EmitMetalSegmentedScanSource(sink); });
}

bool MetalSegmentedScanSourceUpperBytes(std::uint64_t &upper) noexcept {
  return backend_source_recipe::bytes(
      [](auto &sink) noexcept { return EmitMetalSegmentedScanSource(sink); },
      upper);
}

} // namespace rund::node::accel::detail
