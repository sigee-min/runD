#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>

#include "local.hpp"
#include <node/accel/pick.hpp>

#include <kernel/program/compute/dsl.hpp>
#include <node/accel/context.hpp>

#include <memory>
#include <string_view>
#include <type_traits>

namespace node_accel_contract {

bool ContextRejectsInvalidInputs() {
  namespace ctx = node_accel_contract::context;
  static_assert(std::is_same_v<decltype(&rund::node::accel::CreateAccelBuffer),
                               rund::AccelBuffer (*)(const rund::AccelContext &,
                                                     rund::AccelBufferDesc)>);
  static_assert(
      std::is_same_v<decltype(&rund::node::accel::UploadAccelBuffer),
                     rund::AccelCheck (*)(
                         const rund::AccelContext &, const rund::AccelBuffer &,
                         const void *, std::uint64_t, std::uint64_t)>);
  static_assert(
      std::is_same_v<decltype(&rund::node::accel::DownloadAccelBuffer),
                     rund::AccelCheck (*)(const rund::AccelContext &,
                                          const rund::AccelBuffer &, void *,
                                          std::uint64_t, std::uint64_t)>);

  const rund::AccelDevice invalid_pick{};
  if (!ctx::CheckReason(rund::node::accel::OpenAccel(invalid_pick).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  rund::AccelDevice fake =
      rund::node::accel::PickAccel(ctx::Policy(rund::AccelApi::Fake, true));
  if (!fake.check.ok || fake.api != rund::AccelApi::Fake ||
      !ctx::CheckReason(rund::node::accel::OpenAccel(fake).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  rund::AccelDevice forged = fake;
  forged.api = rund::AccelApi::Metal;
  forged.check = rund::AccelCheck{true, "ok"};
  forged.owner = std::make_shared<int>(3);
  forged.backend.context = forged.owner.get();
  forged.backend.execute = nullptr;
  forged.caps.ok = true;
  if (!ctx::CheckReason(rund::node::accel::OpenAccel(forged).check,
                        "accel_context_pick_invalid")) {
    return false;
  }

  const rund::AccelContext invalid_world{};
  const rund::AccelBuffer missing = rund::node::accel::OpenAccelBuffer(
      invalid_world, rund::Buffer{}, ctx::TypedDesc());
  return ctx::CheckReason(missing.check, "accel_context_buffer_invalid") &&
         std::string_view{missing.reason} == "accel_context_buffer_invalid";
}

} // namespace node_accel_contract
