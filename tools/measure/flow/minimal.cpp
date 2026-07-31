#include <rund/compute.hpp>

#include <cstdint>

auto BuildMinimal() {
  return rund::compute::on(rund::compute::Target::cpu(1u))
      .map<std::int32_t>("minimal", 4096u,
                         [](auto value) { return value * 3 + 1; });
}
