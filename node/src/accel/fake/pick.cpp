#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include <memory>

namespace rund::node::accel::detail {

rund::AccelDevice PickFake(const bool allow_fake) {
  if (!allow_fake) {
    return rund::AccelDevice{
        .check = rund::AccelCheck{false, "accel_fake_disallowed"},
        .api = rund::AccelApi::Fake,
    };
  }

  std::shared_ptr<FakeAdapter> adapter = std::make_shared<FakeAdapter>();
  std::shared_ptr<void> owner = adapter;
  return rund::AccelDevice{
      .check = rund::AccelCheck{true, "ok"},
      .api = rund::AccelApi::Fake,
      .caps = adapter->caps,
      .backend =
          rund::kernel::ComputeBackendDispatch{
              .context = adapter.get(),
              .execute = ExecuteFake,
          },
      .owner = owner,
  };
}

} // namespace rund::node::accel::detail
