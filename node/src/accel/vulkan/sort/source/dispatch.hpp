#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Sink>
[[nodiscard]] bool AppendVulkanSortDispatchSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder source{sink};
  source += "layout(set = 0, binding = 0, std430) buffer DispatchArgs {\n";
  source += "  uint dispatch_args[];\n};\n";
  source += "layout(set = 0, binding = 4, std430) buffer Status {\n";
  source += "  uint status[];\n};\n";
  source += "void main() {\n";
  source += "  if (gl_LocalInvocationID.x != 0u) { return; }\n";
  source += "  const SortRange range = ResolveRange();\n";
  source += "  const uint64_t blocks = (range.logical + "
            "uint64_t(kSortBlockSize) - uint64_t(1)) / "
            "uint64_t(kSortBlockSize);\n";
  source += "  for (uint chunk = 0u; chunk < params.chunk_count; ++chunk) {\n";
  source += "    const uint64_t base = uint64_t(chunk) * "
            "uint64_t(params.max_dispatch_groups);\n";
  source += "    const uint groups = range.invalid || blocks <= base ? 0u : "
            "uint(min(uint64_t(params.max_dispatch_groups), blocks - base));\n";
  source += "    dispatch_args[chunk * 3u] = groups;\n";
  source += "    dispatch_args[chunk * 3u + 1u] = 1u;\n";
  source += "    dispatch_args[chunk * 3u + 2u] = 1u;\n";
  source += "  }\n";
  source += "  status[0] = range.invalid ? 2u : 0u;\n";
  source += "}\n";
  return source.valid();
}
#endif

} // namespace rund::node::accel::detail
