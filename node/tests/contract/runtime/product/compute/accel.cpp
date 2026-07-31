#include "accel/local.hpp"

#include <rund/compute/device.hpp>

#include <array>
#include <cstdio>

int RunRuntimeComputeAccelContract() {
  constexpr std::array targets{rund::compute::Target::metal(),
                               rund::compute::Target::vulkan()};
  for (const rund::compute::Target target : targets) {
    const rund::compute::Backend backend = target.backend();
    {
      const auto selected = rund::compute::open(target);
      if (!selected) {
        if (selected.reason() != rund::compute::Reason::AdapterUnavailable) {
          return static_cast<int>(backend) * 10 + 1;
        }
        continue;
      }
    }
    if (const int result =
            rund::node::test_contract::CheckComputeAccelBackend(target);
        result != 0) {
      std::fprintf(stderr, "runtime compute accel backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 10 + result;
    }
  }
  return 0;
}
