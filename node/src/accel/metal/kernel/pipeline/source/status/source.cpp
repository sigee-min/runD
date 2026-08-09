#include "../../source.hpp"
#include "source.hpp"

#include "../../../../../kernel/backend/phase_source.hpp"

namespace rund::node::accel::detail {
namespace {

// Preserve the former fixed recipe's 48-byte preamble, 28,544-byte body, and
// 1,024-byte generated phase-contract allowance.
inline constexpr std::size_t MetalPipelineStatusSourceCapacity =
    48u + 28544u + 1024u;

template <typename Sink>
[[nodiscard]] bool EmitMetalPipelineStatusSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  using namespace metal_pipeline_status_source;
  return sink.append(preamble()) &&
         EmitPipelineNestedPhaseContract(
             sink, PipelineNestedPhaseSourceLanguage::Metal) &&
         sink.append(abi()) && sink.append(reset()) && sink.append(publish()) &&
         sink.append(advance()) && sink.append(reduce());
}

} // namespace

std::string_view MetalPipelineStatusSource() noexcept {
  static const auto source = backend_source_recipe::materialize_fixed<
      MetalPipelineStatusSourceCapacity>(
      [](auto &sink) noexcept { return EmitMetalPipelineStatusSource(sink); });
  return source.text();
}

} // namespace rund::node::accel::detail
