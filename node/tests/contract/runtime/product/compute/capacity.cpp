#include "../support.hpp"
#include "src/runtime/compute/slots.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace rund::node::test_contract {
namespace {

[[nodiscard]] bool CheckSlotSet() {
  runtime_detail::SlotSet slots{};
  slots.configure(130u);
  std::size_t allocated = 0u;
  for (std::size_t index = 0u; index < 130u; ++index) {
    const auto claimed = slots.claim(allocated);
    if (!claimed.has_value() || *claimed != index) {
      return false;
    }
    ++allocated;
  }
  if (slots.claim(allocated).has_value() || !slots.release(64u) ||
      !slots.release(3u) || !slots.release(129u) || slots.release(3u) ||
      slots.claimed(3u)) {
    return false;
  }
  constexpr std::array<std::size_t, 3u> expected{3u, 64u, 129u};
  for (const std::size_t index : expected) {
    const auto claimed = slots.claim(allocated);
    if (!claimed.has_value() || *claimed != index) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckCoordinatorFrameReason() {
  rund::SessionConfig options = Options();
  options.scheduler.coroutine_frame_bytes = 1u;
  ::rund::Session session{};
  const auto opened = session.open(options);
  if (!opened) {
    std::fprintf(stderr, "compute frame contract open reason=%.*s\n",
                 static_cast<int>(opened.error().size()),
                 opened.error().data());
    return false;
  }

  const std::array<std::int32_t, 1u> input{7};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-frame-capacity", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job) {
    return false;
  }
  const compute::Completion result = session.compute(*job).submit().wait();
  const bool valid =
      !result && result.code() == compute::Code::Capacity &&
      result.reason() == compute::Reason::TaskFrameTooLarge &&
      result.error() == std::string_view{"compute_task_frame_too_large"};
  if (!valid) {
    std::fprintf(
        stderr, "compute frame contract result=%u code=%u reason=%.*s\n",
        result ? 1u : 0u, static_cast<unsigned>(result.code()),
        static_cast<int>(result.error().size()), result.error().data());
  }
  return valid && session.close();
}

} // namespace

int CheckComputeTaskCapacity() {
  if (!CheckSlotSet()) {
    return 11;
  }
  if (!CheckCoordinatorFrameReason()) {
    return 12;
  }
  rund::SessionConfig options = Options();
  options.scheduler.task_capacity = 1u;
  ::rund::Session session{};
  if (!session.open(options)) {
    return 1;
  }

  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program = compute::on(compute::Target::cpu(2u))
                     .map<std::int32_t>("node-host-capacity", input.size(),
                                        [](auto value) { return value + 1; })
                     .compile();
  if (!program) {
    return 2;
  }
  auto held_job = program->resident(input);
  auto job = program->resident(input);
  if (!held_job || !job) {
    return 3;
  }
  compute::Submission held = session.compute(*held_job).submit();
  const auto result = session.compute(*job).submit().wait();
  if (result || result.code() != compute::Code::Capacity ||
      result.error() != std::string_view{"compute_task_capacity"}) {
    return 4;
  }
  if (!held.wait()) {
    return 5;
  }
  if (!session.close()) {
    return 6;
  }

  constexpr std::size_t count = 1u << 20u;
  std::vector<std::int32_t> values(count, 5);
  auto long_program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-close", count,
                             [](auto value) { return value * 9 + 1; })
          .compile();
  if (!long_program) {
    return 7;
  }
  auto long_job = long_program->resident(values);
  if (!long_job) {
    return 8;
  }
  compute::Submission retained{};
  {
    ::rund::Session closing{};
    if (!closing.open(Options())) {
      return 9;
    }
    retained = closing.compute(*long_job).submit();
  }
  const auto cancelled = retained.wait();
  return !cancelled &&
                 cancelled.error() == std::string_view{"compute_cancelled"}
             ? 0
             : 10;
}

} // namespace rund::node::test_contract
