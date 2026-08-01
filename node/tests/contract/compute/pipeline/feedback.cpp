#include "local.hpp"

#include <array>
#include <cstddef>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckHostFeedback(rund::compute::Device &device,
                                    const Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t iterations = 4u;
  constexpr std::array<std::int32_t, 4u> seed{1, 3, 5, 7};

  auto input = Upload(device, seed);
  auto output = device.buffer<std::int32_t>(seed.size());
  auto body = on(device)
                  .map<std::int32_t>("pipeline host feedback", seed.size(),
                                     [](auto value) { return value + 1; })
                  .compile();
  if (!input || !output || !body) {
    return 1;
  }
  auto prepared =
      pipeline(device).then(*body, read(*input), write(*output)).prepare();
  if (!prepared) {
    return 2;
  }

  std::array<std::array<std::int32_t, seed.size()>, iterations> observed{};
  std::size_t callbacks = 0u;
  std::uint64_t execution_submits = 0u;
  std::uint64_t feedback_bytes = 0u;
  Reason final_write_reason = Reason::Ok;
  const Status completed = host_feedback(
      *prepared, iterations, [&](HostIteration &iteration) noexcept -> Status {
        if (iteration.completed() != callbacks + 1u ||
            iteration.total() != iterations ||
            iteration.remaining() != iterations - iteration.completed() ||
            iteration.has_next() != (iteration.completed() < iterations)) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const Stats execution = iteration.stats();
        if (!WarmCountersClean(execution) ||
            (backend != Backend::Cpu && execution.command_submits != 1u)) {
          return Status::fail(Reason::CompletionInvalid);
        }
        execution_submits += execution.command_submits;
        Status status = iteration.read(
            *output, std::span<std::int32_t>{observed[callbacks]});
        if (!status) {
          return status;
        }
        ++callbacks;
        if (!iteration.has_next()) {
          final_write_reason =
              iteration
                  .write(*input, std::span<const std::int32_t>{observed.back()})
                  .reason();
          return Status::success();
        }
        status = iteration.write(
            *input, std::span<const std::int32_t>{observed[callbacks - 1u]});
        feedback_bytes += iteration.write_stats().bytes;
        return status;
      });

  constexpr std::array<std::array<std::int32_t, seed.size()>, iterations>
      expected{{
          {2, 4, 6, 8},
          {3, 5, 7, 9},
          {4, 6, 8, 10},
          {5, 7, 9, 11},
      }};
  if (!completed || callbacks != iterations || observed != expected ||
      prepared->generation() != iterations ||
      final_write_reason != Reason::AlreadyCompleted ||
      feedback_bytes !=
          (iterations - 1u) * seed.size() * sizeof(std::int32_t) ||
      (backend != Backend::Cpu && execution_submits != iterations) ||
      (backend == Backend::Cpu && execution_submits != 0u)) {
    return 3;
  }

  const std::uint64_t generation = prepared->generation();
  std::size_t zero_callbacks = 0u;
  const Status zero =
      host_feedback(*prepared, 0u, [&](HostIteration &) noexcept -> Status {
        ++zero_callbacks;
        return Status::success();
      });
  if (!zero || zero_callbacks != 0u || prepared->generation() != generation) {
    return 4;
  }

  auto failure_input = Upload(device, seed);
  auto failure_output = device.buffer<std::int32_t>(seed.size());
  auto failure_pipeline =
      failure_input && failure_output
          ? pipeline(device)
                .then(*body, read(*failure_input), write(*failure_output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!failure_pipeline) {
    return 5;
  }
  std::size_t failure_callbacks = 0u;
  const Status stopped = host_feedback(
      *failure_pipeline, iterations,
      [&](HostIteration &iteration) noexcept -> Status {
        ++failure_callbacks;
        std::array<std::int32_t, seed.size()> values{};
        const Status read_status =
            iteration.read(*failure_output, std::span<std::int32_t>{values});
        if (!read_status) {
          return read_status;
        }
        if (iteration.completed() == 2u) {
          return Status::fail(Reason::Cancelled);
        }
        return iteration.write(*failure_input,
                               std::span<const std::int32_t>{values});
      });
  if (stopped.reason() != Reason::Cancelled || failure_callbacks != 2u ||
      failure_pipeline->generation() != 2u || failure_pipeline->poisoned()) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
