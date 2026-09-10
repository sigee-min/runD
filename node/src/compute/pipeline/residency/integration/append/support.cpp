#include "local.hpp"

#include "../../../../size.hpp"
#include "../../../../type.hpp"

#include <limits>

namespace rund::compute::detail {

[[nodiscard]] PipelineBinding
append_frame_binding(const std::uint32_t owner, const Type type,
                     const FixedFormat format, const std::size_t count,
                     const std::size_t offset, const std::size_t backing_count,
                     const ResourceAccess access) noexcept {
  const std::size_t width = type_bytes(type);
  const std::size_t bytes =
      width != 0u &&
              backing_count <= std::numeric_limits<std::size_t>::max() / width
          ? backing_count * width
          : 0u;
  return PipelineBinding{.type = type,
                         .format = format,
                         .offset = offset,
                         .count = count,
                         .stride = 1u,
                         .element_bytes = width,
                         .alignment = width,
                         .backing_bytes = bytes,
                         .access = access,
                         .owner = owner};
}

} // namespace rund::compute::detail
