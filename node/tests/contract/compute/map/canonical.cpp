#include "local/model.hpp"

#include <rund/compute.hpp>

#include "src/compute/expression/state.hpp"
#include "src/compute/map/build.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace compute_map_contract {
[[nodiscard]] bool Canonical() {
  using namespace rund::compute::detail;
  auto state = make_expr();
  const ExprRef first = input(state, Type::I32, 0u);
  const ExprRef same = input(state, Type::I32, 0u);
  const ExprRef different_type = input(state, Type::U32, 0u);
  const ExprRef different_input = input(state, Type::I32, 1u);
  if (state == nullptr || first.node != 1u || same.node != first.node ||
      different_type.node != 2u || different_input.node != 3u ||
      state->nodes.size() != 3u || state->canonical_nodes != 3u) {
    return false;
  }

  // Projected and forged states populate nodes directly. The first append
  // must rebuild the side index and retain the first canonical occurrence.
  auto rebuilt = std::make_shared<ExprState>();
  const ExprNode duplicate{
      .operation = ExprOp::Input, .type = Type::I32, .left = 0u};
  rebuilt->nodes.push_back(duplicate);
  rebuilt->nodes.push_back(duplicate);
  const ExprRef rebuilt_first = input(rebuilt, Type::I32, 0u);
  const ExprRef rebuilt_new = constant(rebuilt, Type::I32, 7u);
  if (rebuilt_first.node != 1u || rebuilt_new.node != 3u ||
      rebuilt->nodes.size() != 3u || rebuilt->canonical_nodes != 3u) {
    return false;
  }

  auto capacity = make_expr();
  for (std::uint64_t bits = 0u; bits < 1024u; ++bits) {
    const ExprRef value = constant(capacity, Type::I32, bits);
    if (value.node != bits + 1u) {
      return false;
    }
  }
  // At capacity an exact duplicate still resolves before the capacity guard;
  // only a novel node owns ExpressionCapacity.
  const ExprRef at_capacity_duplicate = constant(capacity, Type::I32, 511u);
  const ExprRef overflow = constant(capacity, Type::I32, 1024u);
  return at_capacity_duplicate.node == 512u && overflow.node == 0u &&
         capacity->nodes.size() == 1024u && !capacity->status &&
         capacity->status.reason() == rund::compute::Reason::ExpressionCapacity;
}

[[nodiscard]] bool Replay() {
  using namespace rund::compute::detail;
  const std::array outputs{Type::I32, Type::I32};
  const std::array inputs{Type::I32};

  auto shared_state = make_expr();
  const ExprRef shared_input = input(shared_state, Type::I32, 0u);
  const ExprRef shared_sum =
      binary(ExprOp::Add, shared_input, constant(shared_state, Type::I32, 7u));
  const ExprRef shared_product = binary(ExprOp::Multiply, shared_sum,
                                        constant(shared_state, Type::I32, 3u));
  const std::array shared_roots{shared_product, shared_sum};
  const auto shared =
      build_map_operation_multi(64u, outputs, inputs, shared_roots);

  auto product_state = make_expr();
  const ExprRef product_input = input(product_state, Type::I32, 0u);
  const ExprRef product_sum = binary(ExprOp::Add, product_input,
                                     constant(product_state, Type::I32, 7u));
  const ExprRef product = binary(ExprOp::Multiply, product_sum,
                                 constant(product_state, Type::I32, 3u));
  auto sum_state = make_expr();
  const ExprRef sum = binary(ExprOp::Add, input(sum_state, Type::I32, 0u),
                             constant(sum_state, Type::I32, 7u));
  const std::array distinct_roots{product, sum};
  const auto distinct =
      build_map_operation_multi(64u, outputs, inputs, distinct_roots);

  if (!shared || !distinct) {
    return false;
  }
  const auto shared_map = shared->map();
  const auto distinct_map = distinct->map();
  return shared->ir().canonical_bytes == distinct->ir().canonical_bytes &&
         shared_map.op_hash_hi == distinct_map.op_hash_hi &&
         shared_map.op_hash_lo == distinct_map.op_hash_lo &&
         shared_map.input_bytes_per_tile == distinct_map.input_bytes_per_tile &&
         shared_map.output_bytes_per_tile == distinct_map.output_bytes_per_tile;
}

} // namespace compute_map_contract
