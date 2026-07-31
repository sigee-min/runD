#include "run/local/model.hpp"

int CheckReplayRunContract(const RuntimeReplayFixture &fixture) {
  rund::SessionConfig config = fixture.spec;
  config.scheduler.host_payload_capacity_bytes = 1024u * 1024u;
  config.replay.storage.max_bytes = 1024u * 1024u;

  if (const int result = runtime_task_replay_run::Capacity(config);
      result != 0) {
    return result;
  }

  runtime_task_replay_run::Model model{config};
  if (const int result = runtime_task_replay_run::Surface(model); result != 0) {
    return result;
  }
  if (const int result = runtime_task_replay_run::Scenario(model);
      result != 0) {
    return result;
  }
  if (const int result = runtime_task_replay_run::Lifetime(model);
      result != 0) {
    return result;
  }
  return runtime_task_replay_run::History(model);
}
