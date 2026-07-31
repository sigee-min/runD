#include "local.hpp"
#include "source/block.hpp"
#include "source/offset.hpp"
#include "source/prefix.hpp"
#include "source/prelude.hpp"

namespace rund::node::accel::detail {

std::string MetalSegmentedScanSource() {
  std::string source;
  source += MetalSegmentedPrelude();
  source += MetalSegmentedBlockSource();
  source += MetalSegmentedPrefixSource();
  source += MetalSegmentedOffsetSource();
  return source;
}

} // namespace rund::node::accel::detail
