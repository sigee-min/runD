#include "test/assert.hpp"

#include "../local/model.hpp"

#include <rund/replay.hpp>

#include <filesystem>
#include <vector>

#include <sys/statvfs.h>

namespace replay_spill {

int BudgetContract() {
  const std::filesystem::path root = TempDir("session-shared-budget");
  std::filesystem::remove_all(root);
  TEST_ASSERT(std::filesystem::create_directories(root));
  struct statvfs status{};
  TEST_ASSERT(::statvfs(root.c_str(), &status) == 0);
  const std::uint64_t unit = status.f_frsize != 0u
                                 ? static_cast<std::uint64_t>(status.f_frsize)
                                 : static_cast<std::uint64_t>(status.f_bsize);
  TEST_ASSERT(unit != 0u);

  const ::rund::storage::Budget root_budget{unit};
  rund::SessionConfig config{.workers = 1u};
  config.replay.storage = Storage(root, 1024u);
  config.replay.storage.max_allocated_bytes = unit;
  config.replay.storage.budget = root_budget;
  config.scheduler.host_payload_capacity_bytes =
      config.replay.storage.max_bytes;
  config.replay.input_capacity = 1u;
  rund::Session first_session{};
  rund::Session second_session{};
  TEST_ASSERT(first_session.open(config));
  TEST_ASSERT(second_session.open(config));

  const std::vector<std::byte> payload = Bytes("shared session budget");
  constexpr rund::replay::Input input{.id = 81u, .schema = 83u};
  const rund::replay::Binding binding{};
  auto source = [&](rund::replay::Writer &writer) {
    TEST_ASSERT(writer.append(payload));
    return std::uint64_t{1u};
  };
  const auto channel = binding.input(input, source);
  auto simulate = [&](rund::replay::Context &context) {
    static_cast<void>(channel.read(context));
  };

  {
    const rund::replay::Record held =
        rund::replay::record(first_session, simulate);
    TEST_ASSERT(held);
    TEST_ASSERT(held.storage_report().allocated_bytes == unit);
    TEST_ASSERT(root_budget.report().allocated_bytes == unit);
    const std::uint64_t rejections = root_budget.report().rejection_count;
    const rund::replay::Record rejected =
        rund::replay::record(second_session, simulate);
    TEST_ASSERT(!rejected);
    TEST_ASSERT(root_budget.report().allocated_bytes == unit);
    TEST_ASSERT(root_budget.report().reserved_bytes == 0u);
    TEST_ASSERT(root_budget.report().rejection_count == rejections + 1u);
  }

  TEST_ASSERT(root_budget.report().physical_bytes == 0u);
  TEST_ASSERT(root_budget.report().allocated_bytes == 0u);
  {
    const rund::replay::Record accepted =
        rund::replay::record(second_session, simulate);
    TEST_ASSERT(accepted);
    TEST_ASSERT(accepted.storage_report().allocated_bytes == unit);
    TEST_ASSERT(root_budget.report().allocated_bytes == unit);
    TEST_ASSERT(first_session.close());
    TEST_ASSERT(second_session.close());
    TEST_ASSERT(root_budget.report().allocated_bytes == unit);
    TEST_ASSERT(GenerationDirectories(root).size() == 1u);
  }
  TEST_ASSERT(root_budget.report().physical_bytes == 0u);
  TEST_ASSERT(root_budget.report().allocated_bytes == 0u);
  TEST_ASSERT(GenerationDirectories(root).empty());
  std::filesystem::remove_all(root);
  return 0;
}

} // namespace replay_spill
