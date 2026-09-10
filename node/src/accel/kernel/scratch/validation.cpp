#include "../scratch.hpp"

namespace rund::node::accel::detail {

bool ValidKernelScratch(const KernelScratchLayout &layout,
                        const RunBinds &binds) noexcept {
  if (layout.empty()) {
    return true;
  }
  if (!binds.valid() || binds.refs() == nullptr || binds.handles() == nullptr) {
    return false;
  }
  std::uint64_t prior_slot = 0u;
  for (std::size_t index = 0u; index < layout.size(); ++index) {
    const KernelScratchPage page = layout[index];
    if (page.bytes == 0u || page.slot >= binds.size() ||
        binds.handles()[page.slot] == nullptr ||
        binds.refs()[page.slot].offset_bytes > binds.refs()[page.slot].bytes ||
        page.bytes > binds.refs()[page.slot].bytes -
                         binds.refs()[page.slot].offset_bytes ||
        (index != 0u && page.slot <= prior_slot)) {
      return false;
    }
    prior_slot = page.slot;
  }
  return true;
}

} // namespace rund::node::accel::detail
