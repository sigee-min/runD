#include "local.hpp"

namespace program_compute_contract {

int StencilIdentity() {
  const rund::kernel::StencilDesc desc = U32Stencil();
  const rund::kernel::StencilHash a =
      rund::kernel::HashStencil(desc);
  const rund::kernel::StencilHash b =
      rund::kernel::HashStencil(desc);
  TEST_ASSERT(a.hi == b.hi);
  TEST_ASSERT(a.lo == b.lo);

  rund::kernel::StencilDesc changed = desc;
  changed.element_count += 1u;
  const rund::kernel::StencilHash c =
      rund::kernel::HashStencil(changed);
  TEST_ASSERT(a.hi != c.hi || a.lo != c.lo);

  changed = desc;
  changed.element = rund::kernel::StencilElement::U64;
  const rund::kernel::StencilHash d =
      rund::kernel::HashStencil(changed);
  TEST_ASSERT(a.hi != d.hi || a.lo != d.lo);

  changed = desc;
  changed.radius = 2u;
  const rund::kernel::StencilHash e =
      rund::kernel::HashStencil(changed);
  TEST_ASSERT(a.hi != e.hi || a.lo != e.lo);
  return 0;
}

}  // namespace program_compute_contract
