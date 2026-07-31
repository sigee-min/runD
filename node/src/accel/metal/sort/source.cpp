#include "source.hpp"

#include "source/base.hpp"
#include "source/classify.hpp"
#include "source/count/range.hpp"
#include "source/dispatch.hpp"
#include "source/prefix.hpp"
#include "source/rank.hpp"
#include "source/scatter.hpp"
#include "../../sort/block/metal.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalSortSource(const rund::kernel::u32 block_size) {
  std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;

)MSL";
  source += "#define RUND_SORT_BLOCK_SIZE ";
  source += std::to_string(block_size);
  source += "\n#define RUND_SORT_THREAD_COUNT ";
  source += std::to_string(kMetalSortThreadCount);
  source += "\n#define RUND_SORT_ROUND_COUNT ";
  source += std::to_string(kMetalSortRounds);
  source += "\n#define RUND_SORT_SIMD_WIDTH ";
  source += std::to_string(kMetalSortSimdWidth);
  source += "\n#define RUND_SORT_PACKED_PAIRS ";
  source += std::to_string(kMetalSortPackedPairs);
  source += R"MSL(
#define RUND_SORT_BUCKET_COUNT 256
)MSL";
  source += MetalSortBaseSource();
  source += MetalSortRangeSource();
  source += MetalSortDispatchSource();
  source += MetalSortBlockRankSource();
  source += MetalSortClassifySource();
  source += MetalSortPrefixSource();
  source += MetalSortScatterU32Source();
  source += MetalSortScatterU64Source();
  return source;
}

} // namespace rund::node::accel::detail
