#include "local.hpp"

#include "src/accel/kernel/reset/proof.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace node_accel_contract::reset_contract {
namespace {

using rund::node::accel::detail::reset::Bind;
using rund::node::accel::detail::reset::Commands;
using rund::node::accel::detail::reset::Params;
using rund::node::accel::detail::reset::Payload;
using rund::node::accel::detail::reset::Project;
using rund::node::accel::detail::reset::Prove;
using rund::node::accel::detail::reset::Replacement;
using rund::node::accel::detail::reset::Spec;
using rund::node::accel::detail::reset::WordAddressable;

static_assert(!std::is_aggregate_v<Range>);
static_assert(std::is_trivially_copyable_v<Range>);
static_assert(noexcept(Prove(Spec{}, 0u)));

} // namespace

bool CheckWidthAndLayout() {
  const rund::kernel::ResidentBufferRef source{
      .bytes = 48u,
      .offset_bytes = 4u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
  };
  const Spec dense = Project(source, nullptr);
  const Range dense32 = Prove(dense, 20u);
  const Params params = Bind(dense32, 2u);
  if (!Accepted(dense32) || !dense32.dense() || Payload(dense32) != 16u ||
      params.count != 4u || dense32.end() != 20u || params.base != 2u ||
      params.offset_words != 1u || params.stride_words != 1u ||
      params.element_words != 1u || sizeof(params) != 40u) {
    return false;
  }

  const Spec strided64{
      .offset = 8u,
      .count = 3u,
      .stride = 16u,
      .element = 8u,
  };
  const Range sparse = Prove(strided64, 48u);
  const Params sparse_params = Bind(sparse, 1u);
  if (!Accepted(sparse) || sparse.dense() || Payload(sparse) != 24u ||
      sparse.end() != 48u || sparse_params.offset_words != 2u ||
      sparse_params.stride_words != 4u || sparse_params.element_words != 2u ||
      Commands(513u, 256u) != 3u || Commands(1u, 0u) != 0u) {
    return false;
  }

  const Replacement replacement{.count = 6u, .element = 8u};
  const Spec projected = Project(source, &replacement);
  const Range replaced = Prove(projected, 48u);
  return projected.offset == 0u && projected.count == 6u &&
         projected.stride == 8u && projected.element == 8u &&
         projected.dense() && Accepted(replaced) && replaced.offset() == 0u &&
         replaced.end() == 48u;
}

bool CheckInvalidRanges() {
  constexpr std::uint64_t maximum = UINT64_MAX;
  constexpr std::uint64_t word_limit = UINT32_MAX;
  const Spec offset_overflow{
      .offset = maximum - 3u,
      .count = 1u,
      .stride = 8u,
      .element = 8u,
  };
  const Spec stride_overflow{
      .count = 3u,
      .stride = maximum - 3u,
      .element = 4u,
  };
  const Spec shader_limit{
      .offset = static_cast<std::uint64_t>(UINT32_MAX) * 4u,
      .count = 1u,
      .stride = 4u,
      .element = 4u,
  };
  const Spec shader_strided{
      .offset = shader_limit.offset,
      .count = 1u,
      .stride = 8u,
      .element = 4u,
  };
  const std::uint64_t shader_bytes = shader_limit.offset + 4u;
  const Range shader_range = Prove(shader_limit, shader_bytes);
  const Range shader_strided_range = Prove(shader_strided, shader_bytes);
  const Spec oversized_shader{
      .count = static_cast<std::uint64_t>(UINT32_MAX) + 1u,
      .stride = 4u,
      .element = 4u,
  };
  const std::uint64_t oversized_bytes = oversized_shader.count * 4u;
  const Range oversized_range = Prove(oversized_shader, oversized_bytes);
  return Rejected(
             Prove(Spec{.offset = 2u, .count = 1u, .stride = 4u, .element = 4u},
                   8u)) &&
         Rejected(Prove(Spec{.count = 2u, .stride = 6u, .element = 4u}, 16u)) &&
         Rejected(Prove(Spec{.count = 2u, .stride = 4u, .element = 8u}, 16u)) &&
         Rejected(Prove(offset_overflow, maximum)) &&
         Rejected(Prove(stride_overflow, maximum)) &&
         Rejected(
             Prove(Spec{.offset = 8u, .count = 3u, .stride = 8u, .element = 8u},
                   31u)) &&
         Accepted(shader_range) && shader_range.dense() &&
         Accepted(shader_strided_range) && !shader_strided_range.dense() &&
         Accepted(oversized_range) &&
         !WordAddressable(shader_range, 0u, word_limit) &&
         WordAddressable(shader_range, shader_limit.offset, word_limit) &&
         WordAddressable(shader_strided_range, shader_limit.offset,
                         word_limit) &&
         !WordAddressable(oversized_range, 0u, word_limit) &&
         !WordAddressable(shader_range, shader_limit.offset + 4u, maximum) &&
         !WordAddressable(shader_range, shader_limit.offset - 1u, maximum) &&
         WordAddressable(shader_range, 0u, maximum);
}

} // namespace node_accel_contract::reset_contract
