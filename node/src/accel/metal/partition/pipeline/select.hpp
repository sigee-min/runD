#pragma once

#include "name.hpp"
#include "wide.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct PartitionPipelineNames {
  const char *classify_key;
  const char *classify_function;
  const char *scatter_key;
  const char *scatter_function;
};

[[nodiscard]] inline PartitionPipelineNames
SelectPartitionPipelines(const bool wide_flags,
                         const bool wide_values) noexcept {
  return {
      wide_flags ? kPartitionClassifyU64Key : kPartitionClassifyKey,
      wide_flags ? kPartitionClassifyU64Function : kPartitionClassifyFunction,
      wide_flags
          ? (wide_values ? kPartitionScatterF64V64Key
                         : kPartitionScatterF64V32Key)
          : (wide_values ? kPartitionScatterU64Key : kPartitionScatterU32Key),
      wide_flags ? (wide_values ? kPartitionScatterF64V64Function
                                : kPartitionScatterF64V32Function)
                 : (wide_values ? kPartitionScatterU64Function
                                : kPartitionScatterU32Function),
  };
}
#endif

} // namespace rund::node::accel::detail
