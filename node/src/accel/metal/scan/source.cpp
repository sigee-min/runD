#include "source.hpp"

#include "source/base.hpp"
#include "source/32/program.hpp"
#include "source/64/program.hpp"

namespace rund::node::accel::detail {

std::string MetalScanSource() {
  std::string source;
  source += MetalScanBaseSource();
  source += MetalScanBlockU32Source();
  source += MetalScanBlockFlagU32Source();
  source += MetalScanPrefixU32Source();
  source += MetalScanOffsetU32Source();
  source += MetalScanBlockU64Source();
  source += MetalScanPrefixU64Source();
  source += MetalScanOffsetU64Source();
  return source;
}

} // namespace rund::node::accel::detail
