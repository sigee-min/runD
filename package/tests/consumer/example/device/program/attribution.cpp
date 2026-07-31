#include "model.hpp"

namespace package_device_program {

[[nodiscard]] int CheckAttribution(Device &device, const Backend backend) {
  constexpr std::array<std::int32_t, 8u> input{1, 2, 3, 4, 5, 6, 7, 8};
  auto broadphase = on(device)
                        .map<std::int32_t>("consumer-broadphase", input.size(),
                                           [](auto value) { return value + 1; })
                        .compile();
  auto contacts =
      on(device)
          .map<std::int32_t>("consumer-contact-generation", input.size(),
                             [](auto value) { return value * 2; })
          .compile();
  auto solver = on(device)
                    .map<std::int32_t>("consumer-solver", input.size(),
                                       [](auto value) { return value - 3; })
                    .compile();
  auto publication =
      on(device)
          .map<std::int32_t>("consumer-publication", input.size(),
                             [](auto value) { return value; })
          .compile();
  if (!broadphase || !contacts || !solver || !publication) {
    return 1;
  }
  const std::array programs{broadphase->fingerprint(), contacts->fingerprint(),
                            solver->fingerprint(), publication->fingerprint()};
  for (std::size_t left = 0u; left < programs.size(); ++left) {
    for (std::size_t right = left + 1u; right < programs.size(); ++right) {
      if (programs[left] == programs[right]) {
        return 2;
      }
    }
  }

  auto baseline_input = device.upload<std::int32_t>(input);
  auto baseline_bounds = device.buffer<std::int32_t>(input.size());
  auto baseline_contacts = device.buffer<std::int32_t>(input.size());
  auto baseline_solved = device.buffer<std::int32_t>(input.size());
  auto baseline_output = device.buffer<std::int32_t>(input.size());
  auto profiled_input = device.upload<std::int32_t>(input);
  auto profiled_bounds = device.buffer<std::int32_t>(input.size());
  auto profiled_contacts = device.buffer<std::int32_t>(input.size());
  auto profiled_solved = device.buffer<std::int32_t>(input.size());
  auto profiled_output = device.buffer<std::int32_t>(input.size());
  if (!baseline_input || !baseline_bounds || !baseline_contacts ||
      !baseline_solved || !baseline_output || !profiled_input ||
      !profiled_bounds || !profiled_contacts || !profiled_solved ||
      !profiled_output) {
    return 3;
  }
  auto baseline =
      pipeline(device)
          .then(*broadphase, read(*baseline_input), write(*baseline_bounds))
          .then(*contacts, read(*baseline_bounds), write(*baseline_contacts))
          .then(*solver, read(*baseline_contacts), write(*baseline_solved))
          .then(*publication, read(*baseline_solved), write(*baseline_output))
          .prepare();
  auto profiled =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .then(*broadphase, read(*profiled_input), write(*profiled_bounds))
          .then(*contacts, read(*profiled_bounds), write(*profiled_contacts))
          .then(*solver, read(*profiled_contacts), write(*profiled_solved))
          .then(*publication, read(*profiled_solved), write(*profiled_output))
          .prepare();
  if (!baseline || !profiled) {
    return 4;
  }
  const Status baseline_run = baseline->run();
  const Status profiled_run = profiled->run();
  const Stats baseline_stats = baseline->stats();
  const Stats profiled_stats = profiled->stats();
  std::array<PipelineStepProfile, 4u> rows{};
  const auto profile = profiled->profile(rows);
  if (!baseline_run || !profiled_run || !profile ||
      baseline->fingerprint() != profiled->fingerprint() ||
      baseline_run.reason() != profiled_run.reason() ||
      baseline_stats.backend != backend || profiled_stats.backend != backend ||
      baseline_stats.command_submits != profiled_stats.command_submits ||
      baseline_stats.dispatches != profiled_stats.dispatches ||
      baseline_stats.pipeline_compiles != 0u ||
      profiled_stats.pipeline_compiles != 0u ||
      baseline_stats.buffer_allocations != 0u ||
      profiled_stats.buffer_allocations != 0u ||
      baseline_stats.descriptor_set_allocations != 0u ||
      profiled_stats.descriptor_set_allocations != 0u ||
      !valid_profile(backend, *profile, rows, programs, profiled->memory(),
                     sizeof(std::int32_t) * input.size() * 5u)) {
    return 5;
  }
  if (backend != Backend::Cpu) {
    for (const PipelineStepProfile &row : rows) {
      if (row.execution.workgroup_count == 0u ||
          row.execution.work_item_count != input.size()) {
        return 6;
      }
    }
  }
  std::array<std::int32_t, input.size()> baseline_values{};
  std::array<std::int32_t, input.size()> profiled_values{};
  const Status baseline_read =
      baseline->read(*baseline_output, baseline_values);
  const Status profiled_read =
      profiled->read(*profiled_output, profiled_values);
  constexpr std::array<std::int32_t, input.size()> expected{1, 3,  5,  7,
                                                            9, 11, 13, 15};
  return baseline_read && profiled_read && baseline_values == expected &&
                 profiled_values == expected
             ? 0
             : 7;
}

} // namespace package_device_program
