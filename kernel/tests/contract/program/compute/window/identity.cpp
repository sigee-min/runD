#include "local.hpp"

namespace program_compute_contract {
namespace {

[[nodiscard]] bool Different(const rund::kernel::WindowHash left,
                             const rund::kernel::WindowHash right) noexcept {
  return left.hi != right.hi || left.lo != right.lo;
}

} // namespace

int WindowIdentity() {
  const rund::kernel::WindowDesc desc = U32Window();
  const rund::kernel::WindowHash original = rund::kernel::HashWindow(desc);
  const rund::kernel::WindowHash repeated = rund::kernel::HashWindow(desc);
  TEST_ASSERT(!Different(original, repeated));

  rund::kernel::WindowDesc changed = desc;
  changed.op = rund::kernel::WindowOp::Min;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  changed.element = rund::kernel::WindowElement::U64;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  changed.boundary = rund::kernel::WindowBoundary::Clip;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  changed.domain = rund::kernel::ComputeDomain::I32;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  changed.domain = rund::kernel::ComputeDomain::Fixed;
  changed.fixed_format = Fixed32(rund::kernel::ComputeOverflow::Wrap);
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed.fixed_format.overflow = rund::kernel::ComputeOverflow::Saturate;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));

  changed = desc;
  changed.count_source = rund::kernel::ComputeCountSource::BufferU32;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));

  changed = desc;
  ++changed.input_count;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  ++changed.output_count;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  ++changed.window_size;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  ++changed.stride;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  changed = desc;
  ++changed.pad_left;
  TEST_ASSERT(Different(original, rund::kernel::HashWindow(changed)));
  return 0;
}

} // namespace program_compute_contract
