#include <accel/api.hpp>
#include <accel/device.hpp>

#include "compile/local.hpp"
#include "test/assert.hpp"
#include <node/accel/pick.hpp>

namespace node_accel_contract {

bool AccelGraphKernelCompileContract() {
  namespace c = kernel_case::compile;
  namespace k = kernel_case;
  TEST_ASSERT(c::SignatureAndSupportRejects());

  rund::AccelDevice pick =
      rund::node::accel::PickAccel(k::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    if (!k::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Metal)) {
      return false;
    }
    pick = rund::node::accel::PickAccel(k::Policy(rund::AccelApi::Vulkan));
  }
  if (!pick.check.ok) {
    return k::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Vulkan);
  }

  c::Fixture fixture = c::MakeFixture(pick);
  TEST_ASSERT(c::Prepare(fixture));
  TEST_ASSERT(c::IdentityIsStable(fixture));
  TEST_ASSERT(c::InitializationIsGraphOwned(fixture));
  TEST_ASSERT(c::GraphIdIgnoresBufferIds(fixture));
  TEST_ASSERT(c::MultiNodeGraphIdMatchesKernel(fixture));
  TEST_ASSERT(c::TwoReadBindingOrderRejects(fixture));
  TEST_ASSERT(c::TamperedSupportRejects(fixture));
  TEST_ASSERT(c::ForeignBufferRejects(fixture));
  TEST_ASSERT(c::LogicalBufferIdentityRejects(fixture));
  TEST_ASSERT(c::UnsupportedAndInvalidGraphRejects(fixture));
  return true;
}

} // namespace node_accel_contract
