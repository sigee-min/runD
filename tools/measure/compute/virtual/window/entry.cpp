#include "local.hpp"

#include "../window.hpp"

#include <cstdio>

namespace rund::measure::compute {

bool MeasureVirtualWindow() {
  auto cpu = virtual_window::prepare(Backend::Cpu);
  auto metal = virtual_window::prepare(Backend::Metal);
  if (cpu == nullptr || metal == nullptr) {
    std::fputs("virtual window paired preparation failed\n", stderr);
    return false;
  }
  for (std::size_t index = 0u; index < virtual_window::ActiveCounts.size();
       ++index) {
    if (!virtual_window::measure_cell(*cpu, *metal,
                                      virtual_window::ActiveCounts[index],
                                      virtual_window::EpochCounts[index])) {
      std::fprintf(
          stderr, "virtual window q=%llu evidence failed\n",
          static_cast<unsigned long long>(virtual_window::EpochCounts[index]));
      return false;
    }
  }
  return true;
}

} // namespace rund::measure::compute
