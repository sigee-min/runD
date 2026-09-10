#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

bool project_residency_transform_uses(
    const VirtualEpochProjection &epoch, const VirtualRunProjection &run,
    const std::span<residency::CacheUse> input_uses,
    const std::span<residency::CacheUse> output_uses) noexcept {
  if (epoch.page_count == 0u || input_uses.size() != epoch.page_count ||
      output_uses.size() != epoch.page_count) {
    return false;
  }
  for (std::size_t index = 0u; index < input_uses.size(); ++index) {
    if (index > std::numeric_limits<std::uint64_t>::max() - epoch.failed_page) {
      return false;
    }
    const std::uint64_t page = epoch.failed_page + index;
    std::uint64_t next_use = residency::NeverUse;
    residency::DirtyExtent dirty{};
    if (!run.active.stream.next_use(page, next_use) ||
        !project_virtual_output_dirty(run, page, dirty)) {
      return false;
    }
    input_uses[index] = residency::CacheUse{
        .key =
            virtual_cache_key(run, run.input_backing, run.input_version, page),
        .access = residency::Access::Read,
        .next_use = next_use,
    };
    output_uses[index] = residency::CacheUse{
        .key = virtual_output_cache_key(run, run.output_backing,
                                        run.output_version, page),
        .access = residency::Access::Write,
        .next_use = residency::NeverUse,
        .dirty = dirty,
    };
  }
  return true;
}

} // namespace rund::compute::detail
