#include <accel/graph/node.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

void PopulateBaseCompileData(const rund::AccelGraphNode &node,
                             GraphCompileNode &compile_data) noexcept {
  compile_data.ir = node.ir;
  compile_data.primitive_hash_hi = node.primitive_hash_hi;
  compile_data.primitive_hash_lo = node.primitive_hash_lo;
  compile_data.element_count = node.element_count;
  compile_data.control = node.control;
  compile_data.signature = node.signature;
}

} // namespace rund::node::accel::detail
