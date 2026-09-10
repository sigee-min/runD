#include "source.hpp"

#include "source/common.hpp"
#include "source/finalize.hpp"
#include "source/reduce.hpp"

#include "../../../../kernel/backend/phase/source.hpp"
#include "../../../../kernel/backend/source/storage.hpp"

#include <cstddef>

namespace rund::node::accel::detail {
namespace {

// The aggregate source is fixed after the phase contract is emitted. Keep the
// capacity explicit so all fragments share one bounded publication owner.
inline constexpr std::size_t MetalNestedAggregateSourceCapacity = 32768u;

template <typename Sink>
[[nodiscard]] bool EmitMetalNestedAggregateSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  return sink.append(MetalNestedAggregatePreambleSource()) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Metal) &&
         sink.append(MetalNestedAggregateCommonSource()) &&
         sink.append(MetalNestedAggregateReduceSource()) &&
         sink.append(MetalNestedAggregateFinalizeSource());
}

} // namespace

std::string_view MetalNestedAggregateSource() noexcept {
  static const auto source = backend_source_recipe::materialize_fixed<
      MetalNestedAggregateSourceCapacity>(
      [](auto &sink) noexcept { return EmitMetalNestedAggregateSource(sink); });
  return source.text();
}

} // namespace rund::node::accel::detail
