#include "local.hpp"
namespace program_compute_contract {

int ReduceIdentity() {
  const rund::kernel::ReduceDesc desc = U32Reduce();
  const rund::kernel::ReduceHash first = rund::kernel::HashReduce(desc);
  const rund::kernel::ReduceHash second = rund::kernel::HashReduce(desc);
  rund::kernel::ReduceDesc changed = desc;
  changed.element_count = 1026u;
  const rund::kernel::ReduceHash changed_hash = rund::kernel::HashReduce(changed);
  rund::kernel::ReduceDesc changed_op = desc;
  changed_op.op = rund::kernel::ReduceOp::CountNonzero;
  const rund::kernel::ReduceHash changed_op_hash =
      rund::kernel::HashReduce(changed_op);
  rund::kernel::ReduceDesc min_op = desc;
  min_op.op = rund::kernel::ReduceOp::Min;
  const rund::kernel::ReduceHash min_op_hash =
      rund::kernel::HashReduce(min_op);
  rund::kernel::ReduceDesc max_op = desc;
  max_op.op = rund::kernel::ReduceOp::Max;
  const rund::kernel::ReduceHash max_op_hash =
      rund::kernel::HashReduce(max_op);
  rund::kernel::ReduceDesc bounded = desc;
  bounded.count_source = rund::kernel::ComputeCountSource::BufferU64;
  const rund::kernel::ReduceHash bounded_hash =
      rund::kernel::HashReduce(bounded);
  rund::kernel::ReduceHash descriptor_only{
      .hi = 0x452821e638d01377ull,
      .lo = 0xbe5466cf34e90c6cull,
  };
  descriptor_only = rund::kernel::reduce_identity_detail::Mix(
      descriptor_only, static_cast<rund::kernel::u64>(desc.op));
  descriptor_only = rund::kernel::reduce_identity_detail::Mix(
      descriptor_only, static_cast<rund::kernel::u64>(desc.element));
  descriptor_only = rund::kernel::reduce_identity_detail::Mix(
      descriptor_only, desc.element_count);
  descriptor_only = rund::kernel::reduce_identity_detail::Mix(
      descriptor_only, desc.block_size);
  descriptor_only = rund::kernel::reduce_identity_detail::Mix(
      descriptor_only,
      static_cast<rund::kernel::u64>(desc.count_source));

  TEST_ASSERT(first.hi == second.hi);
  TEST_ASSERT(first.lo == second.lo);
  TEST_ASSERT(first.hi != 0u || first.lo != 0u);
  TEST_ASSERT(first.hi != changed_hash.hi || first.lo != changed_hash.lo);
  TEST_ASSERT(first.hi != changed_op_hash.hi ||
              first.lo != changed_op_hash.lo);
  TEST_ASSERT(first.hi != min_op_hash.hi || first.lo != min_op_hash.lo);
  TEST_ASSERT(first.hi != max_op_hash.hi || first.lo != max_op_hash.lo);
  TEST_ASSERT(first.hi != bounded_hash.hi || first.lo != bounded_hash.lo);
  TEST_ASSERT(first.hi != descriptor_only.hi ||
              first.lo != descriptor_only.lo);
  return 0;
}

} // namespace program_compute_contract
