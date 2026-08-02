#include "source.hpp"

#include "source/base.hpp"
#include "source/classify.hpp"
#include "source/count/range.hpp"
#include "source/dispatch.hpp"
#include "source/prefix.hpp"
#include "source/rank.hpp"
#include "source/scatter.hpp"
#include "../../sort/block/metal.hpp"
#include "../../kernel/backend/source_recipe.hpp"

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

namespace {
inline constexpr std::string_view Prelude = R"MSL(
#include <metal_stdlib>
using namespace metal;

)MSL";
inline constexpr std::string_view BlockDefine =
    "#define RUND_SORT_BLOCK_SIZE ";
inline constexpr std::string_view ThreadDefine =
    "\n#define RUND_SORT_THREAD_COUNT ";
inline constexpr std::string_view RoundDefine =
    "\n#define RUND_SORT_ROUND_COUNT ";
inline constexpr std::string_view SimdDefine =
    "\n#define RUND_SORT_SIMD_WIDTH ";
inline constexpr std::string_view PackedDefine =
    "\n#define RUND_SORT_PACKED_PAIRS ";
inline constexpr std::string_view BucketDefine = R"MSL(
#define RUND_SORT_BUCKET_COUNT 256
)MSL";

template <typename Sink>
[[nodiscard]] bool EmitMetalSortSource(Sink &sink,
                                       const rund::kernel::u32 block_size) {
  return sink.append(Prelude) && sink.append(BlockDefine) &&
         backend_source_recipe::append_decimal(sink, block_size) &&
         sink.append(ThreadDefine) && backend_source_recipe::append_decimal(
                                          sink, kMetalSortThreadCount) &&
         sink.append(RoundDefine) &&
         backend_source_recipe::append_decimal(sink, kMetalSortRounds) &&
         sink.append(SimdDefine) &&
         backend_source_recipe::append_decimal(sink, kMetalSortSimdWidth) &&
         sink.append(PackedDefine) && backend_source_recipe::append_decimal(
                                          sink, kMetalSortPackedPairs) &&
         sink.append(BucketDefine) && sink.append(MetalSortBaseSource()) &&
         sink.append(MetalSortRangeSource()) &&
         sink.append(MetalSortDispatchSource()) &&
         sink.append(MetalSortBlockRankSource()) &&
         sink.append(MetalSortClassifySource()) &&
         sink.append(MetalSortPrefixSource()) &&
         sink.append(MetalSortScatterU32Source()) &&
         sink.append(MetalSortScatterU64Source());
}
} // namespace

std::string MetalSortSource(const rund::kernel::u32 block_size) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitMetalSortSource(sink, block_size);
  });
}

bool MetalSortSourceUpperBytes(const rund::kernel::u32 block_size,
                               std::uint64_t &upper) noexcept {
  return backend_source_recipe::bytes(
      [&](auto &sink) noexcept {
        return EmitMetalSortSource(sink, block_size);
      },
      upper);
}

} // namespace rund::node::accel::detail
