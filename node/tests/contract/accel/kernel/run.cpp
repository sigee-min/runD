#include <accel/api.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>

#include "run/local.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <iostream>

namespace node_accel_contract {

bool AccelGraphKernelResidentRunContract() {
  namespace k = kernel_case;
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

  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  const k::ResidentRunFixture fixture = k::MakeResidentRunFixture(context);
  const auto require = [](const bool ok, const char *const name) {
    if (!ok) {
      std::cerr << "resident run failure: " << name << '\n';
    }
    return ok;
  };
  if (!require(fixture.ok, "fixture") ||
      !require(k::ResidentRunEvidenceMatches(fixture), "evidence") ||
      !require(k::MultiWriteResidentRunMatches(context), "multi-write") ||
      !require(k::WindowedResidentRunMatches(context), "windowed") ||
      !require(k::ResidentRunRejectsForgedAndForeign(fixture),
               "forged-or-foreign-rejection") ||
      !require(k::LogicalAliasAdmissionMatches(fixture),
               "logical-alias-admission") ||
      !require(k::IndexedWriteBoundaryMatches(context),
               "indexed-write-boundary")) {
    return false;
  }

  constexpr std::array APIs{rund::AccelApi::Metal, rund::AccelApi::Vulkan};
  for (const rund::AccelApi api : APIs) {
    if (api == context.api) {
      continue;
    }
    const rund::AccelDevice alternate =
        rund::node::accel::PickAccel(k::Policy(api));
    if (!alternate.check.ok) {
      if (!k::PickUnavailableReasonIsPrecise(alternate, api)) {
        return false;
      }
      continue;
    }
    const rund::AccelContext alternate_context =
        rund::node::accel::OpenAccel(alternate);
    if (!alternate_context.check.ok ||
        !require(k::IndexedWriteBoundaryMatches(alternate_context),
                 "alternate-indexed-write-boundary")) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
