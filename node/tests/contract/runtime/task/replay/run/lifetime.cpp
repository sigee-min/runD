#include "local/model.hpp"

#include "test/assert.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace runtime_task_replay_run {

int Lifetime(Model &model) {
  std::unique_ptr<rund::replay::Value> escaped{};
  TEST_ASSERT(
      rund::replay::live(model.session, [&](rund::replay::Context &input) {
        escaped.reset(new rund::replay::Value(model.commands.read(input)));
        TEST_ASSERT(*escaped);
      }));
  TEST_ASSERT(escaped != nullptr);
  TEST_ASSERT(!*escaped);
  TEST_ASSERT(escaped->code() == rund::replay::Code::ScopeExpired);
  TEST_ASSERT(escaped->bytes().empty());

  const std::vector<std::byte> artifact = SaveReplayArtifact(*model.baseline);
  const rund::replay::Load<rund::replay::Record> decoded =
      rund::replay::Record::load(artifact);
  TEST_ASSERT(decoded);
  TEST_ASSERT(rund::replay::check(*model.baseline, *decoded));
  rund::replay::Load<rund::replay::Record> record_copy = decoded;
  rund::replay::Load<rund::replay::Record> record_moved =
      std::move(record_copy);
  TEST_ASSERT(record_moved);
  TEST_ASSERT(record_moved->hash() == decoded->hash());
  TEST_ASSERT(record_copy.code() == rund::replay::Code::RecordLoadMovedFrom);
  return 0;
}

} // namespace runtime_task_replay_run
