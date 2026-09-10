#include "source.hpp"

#include "../../kernel/backend/source/storage.hpp"
#include "source/block.hpp"
#include "source/flag.hpp"
#include "source/offset.hpp"
#include "source/prefix.hpp"
#include "source/base.hpp"

namespace rund::node::accel::detail {

namespace {
template <typename Sink> [[nodiscard]] bool EmitMetalScanSource(Sink &sink) {
  return AppendMetalScanBaseSource(sink) &&
         AppendMetalScanBlockSource<false>(sink) &&
         sink.append(MetalScanBlockFlagU32Source()) &&
         AppendMetalScanPrefixSource<false>(sink) &&
         AppendMetalScanOffsetSource<false>(sink) &&
         AppendMetalScanBlockSource<true>(sink) &&
         AppendMetalScanPrefixSource<true>(sink) &&
         AppendMetalScanOffsetSource<true>(sink);
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
