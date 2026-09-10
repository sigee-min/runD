#include "model.hpp"

#include "replay/model.hpp"

namespace package_blackbox {

[[nodiscard]] int CheckRunReplay() {
  const rund::SessionConfig config{
      .workers = 1u,
      .scheduler =
          {
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
              .observation_capacity = 8u,
          },
  };
  TaskRun run_task{};
  const rund::Session::Result result =
      rund::run(config, [&] { RunBlackboxTask(run_task); });
  if (!result) {
    return result.exit_code();
  }
  if (!run_task.joined) {
    return run_task.joined.exit_code();
  }
  if (!run_task.operation) {
    return run_task.operation.exit_code();
  }
  if (!result.ok() || run_task.ran != 1u || result.tasks().spawned() != 1u ||
      result.tasks().completed() != 1u) {
    return Mismatch("run");
  }

  rund::Session replay_session{};
  const auto replay_opened = replay_session.open(config);
  if (!replay_opened) {
    return replay_opened.exit_code();
  }

  return Finish(replay_session, [&]() -> int {
    replay_detail::Fixture fixture{replay_session};
    if (const int result = replay_detail::CheckRecord(fixture); result != 0) {
      return result;
    }
    if (const int result = replay_detail::CheckCodec(fixture); result != 0) {
      return result;
    }
    if (const int result = replay_detail::CheckScenario(fixture); result != 0) {
      return result;
    }
    if (const int result = replay_detail::CheckCheckpoint(fixture);
        result != 0) {
      return result;
    }
    return replay_detail::CheckHistory(fixture);
  });
}

} // namespace package_blackbox
