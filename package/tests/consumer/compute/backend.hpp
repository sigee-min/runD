#pragma once

#include <rund/compute.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace package_compute {

inline int BackendSurface() {
  constexpr std::array targets{
      rund::compute::Target::cpu(),
      rund::compute::Target::metal(),
      rund::compute::Target::vulkan(),
  };
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  const std::vector<std::int32_t> expected{3, 5, 7, 9};

  for (const rund::compute::Target target : targets) {
    const rund::compute::Backend backend = target.backend();
    auto device = rund::compute::open(target);
    if (!device) {
      if (backend == rund::compute::Backend::Cpu) {
        return device.exit_code();
      }
      if (device.error().empty() ||
          (device.code() != rund::compute::Code::Unsupported &&
           device.code() != rund::compute::Code::Unavailable)) {
        return 2;
      }
      continue;
    }
    const auto selected = device->backend();
    if (!selected) {
      return selected.exit_code();
    }
    if (*selected != backend) {
      return 2;
    }
    auto info = device->info();
    if (!info) {
      return info.exit_code();
    }
    if (info->backend != backend || info->storage_alignment == 0u ||
        info->storage_bytes == 0u ||
        (backend == rund::compute::Backend::Cpu &&
         (info->name.empty() || info->driver.empty() ||
          info->driver_details.empty()))) {
      return 2;
    }

    auto program =
        rund::compute::on(*device)
            .map<std::int32_t>("package-backend", input.size(),
                               [](auto value) { return value * 2 + 1; })
            .compile();
    if (!program) {
      return program.exit_code();
    }
    const auto program_backend = program->backend();
    if (!program_backend) {
      return program_backend.exit_code();
    }
    if (*program_backend != backend) {
      return 2;
    }
    auto output = program->run(std::span<const std::int32_t>{input});
    if (!output) {
      return output.exit_code();
    }
    if (*output != expected) {
      return 2;
    }
    if (backend == rund::compute::Backend::Cpu) {
      auto live_program = std::move(*program);
      if (!live_program || program->valid()) {
        return 2;
      }
      const auto moved_program_backend = program->backend();
      if (moved_program_backend || moved_program_backend.reason() !=
                                       rund::compute::Reason::ProgramInvalid) {
        return 2;
      }
      const auto moved_program_run =
          program->run(std::span<const std::int32_t>{input});
      if (moved_program_run ||
          moved_program_run.reason() != rund::compute::Reason::ProgramInvalid) {
        return 2;
      }
    }
  }

  auto opened = rund::compute::open(rund::compute::Target::cpu(1u));
  if (!opened) {
    return opened.exit_code();
  }
  rund::compute::Device live = std::move(*opened);
  if (!live || opened->valid()) {
    return 2;
  }
  const auto moved_backend = opened->backend();
  if (moved_backend ||
      moved_backend.reason() != rund::compute::Reason::DeviceInvalid) {
    return 2;
  }
  const auto rejected = rund::compute::on(*opened)
                            .map<std::int32_t>("package-moved-device", 1u,
                                               [](auto value) { return value; })
                            .compile();
  if (rejected || rejected.reason() != rund::compute::Reason::DeviceInvalid) {
    return 2;
  }
  return 0;
}

} // namespace package_compute
