#include "prelude.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

template <typename Sink>
[[nodiscard]] bool EmitPrelude(Sink &source) noexcept(
    noexcept(source += std::string_view{})) {
  source += "#include <metal_stdlib>\n"
            "using namespace metal;\n";
  if (!AppendSegmentedReduceShaderModel(source)) {
    return false;
  }
  source += R"MSL(
struct RundSegmentedReduceParams {
  ulong count;
  ulong block_count;
  ulong segments_per_group;
};)MSL";
  return source.valid();
}

} // namespace

bool EmitMetalSegmentedReducePreludeSource(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitPrelude(sink);
}

bool EmitMetalSegmentedReducePreludeSource(
    backend_source_recipe::StringSink &sink) {
  return EmitPrelude(sink);
}

#endif

} // namespace rund::node::accel::detail
