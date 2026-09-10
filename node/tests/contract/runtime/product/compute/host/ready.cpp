#include "local.hpp"

#include "../../../../compute/allocation.hpp"
#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>

namespace runtime_compute_host_detail {
namespace {

struct CommitToken final {
  std::atomic<std::uint32_t> *completed = nullptr;
  std::atomic<std::uint32_t> *commit_index = nullptr;
  std::array<std::uint32_t, 4u> *commit_order = nullptr;
  std::uint32_t value = 0u;
  bool armed = true;

  CommitToken(std::atomic<std::uint32_t> &completed_value,
              std::atomic<std::uint32_t> &commit_index_value,
              std::array<std::uint32_t, 4u> &commit_order_value,
              const std::uint32_t task_value) noexcept
      : completed(&completed_value), commit_index(&commit_index_value),
        commit_order(&commit_order_value), value(task_value) {}
  CommitToken(const CommitToken &) = delete;
  CommitToken &operator=(const CommitToken &) = delete;
  CommitToken(CommitToken &&other) noexcept
      : completed(other.completed), commit_index(other.commit_index),
        commit_order(other.commit_order), value(other.value),
        armed(other.armed) {
    other.armed = false;
  }
  CommitToken &operator=(CommitToken &&) = delete;
  ~CommitToken() {
    if (!armed) {
      return;
    }
    const std::uint32_t index =
        commit_index->fetch_add(1u, std::memory_order_relaxed);
    if (index < commit_order->size()) {
      (*commit_order)[index] = value;
    }
  }

  void operator()() const noexcept {
    completed->fetch_add(1u, std::memory_order_relaxed);
  }
};

[[nodiscard]] bool WarmReadyOrder() {
  std::array<rund::task::Handle, 4u> handles{};
  for (rund::task::Handle &handle : handles) {
    handle = rund::task::spawn("ready-order-warm", [] {});
  }
  return static_cast<bool>(rund::task::join_all(handles));
}

} // namespace

[[nodiscard]] ReadyOrder CheckReadyOrder(const std::uint32_t workers) {
  constexpr std::array<std::int32_t, 1u> input{7};
  auto program = rund::compute::on(rund::compute::Target::cpu(2u))
                     .map<std::int32_t>("ready-order", input.size(),
                                        [](auto value) { return value + 1; })
                     .compile();
  if (!program) {
    return {};
  }
  auto job = program->resident(input);
  if (!job) {
    return {};
  }

  rund::SessionConfig options = rund::node::test_contract::Options();
  options.id = 93u;
  options.scheduler.task_workers = workers;
  options.scheduler.task_capacity = 8u;
  options.scheduler.ready_queue_capacity = 8u;
  rund::Session session{};
  if (!session.open(options)) {
    return {};
  }
  if (!session.compute(*job).submit().wait()) {
    static_cast<void>(session.close());
    return {};
  }

  std::atomic<std::uint32_t> completed{0u};
  std::atomic<std::uint32_t> commit_index{0u};
  std::array<std::uint32_t, 4u> commit_order{};
  rund::task::Status joined{};
  bool computed = false;
  std::uint64_t allocations = 0u;
  const rund::Session::Result report = session.scope([&] {
    if (!WarmReadyOrder()) {
      return;
    }
    std::array<rund::task::Handle, 4u> handles{};
    for (std::size_t index = 0u; index < handles.size(); ++index) {
      handles[index] = rund::task::spawn(
          "ready-order", CommitToken{completed, commit_index, commit_order,
                                     static_cast<std::uint32_t>(index + 1u)});
    }
    node_compute_allocation::Start();
    const rund::compute::Completion completion =
        session.compute(*job).submit().wait();
    computed = static_cast<bool>(completion);
    node_compute_allocation::Stop();
    allocations = node_compute_allocation::Count();
    if (!computed) {
      std::fprintf(stderr, "ready order compute: %.*s\n",
                   static_cast<int>(completion.error().size()),
                   completion.error().data());
    }
    joined = rund::task::join_all(handles);
  });
  const bool closed = session.close().ok();
  constexpr std::array<std::uint32_t, 4u> expected{1u, 2u, 3u, 4u};
  const std::uint32_t completed_count =
      completed.load(std::memory_order_relaxed);
  const std::uint32_t committed_count =
      commit_index.load(std::memory_order_relaxed);
  const bool ok = report && computed && joined && closed && allocations == 0u &&
                  completed_count == 4u && committed_count == 4u &&
                  commit_order == expected;
  if (!ok) {
    std::fprintf(
        stderr,
        "ready order workers=%u report=%u computed=%u joined=%u closed=%u "
        "allocations=%llu completed=%u committed=%u order=%u,%u,%u,%u\n",
        workers, report ? 1u : 0u, computed ? 1u : 0u, joined ? 1u : 0u,
        closed ? 1u : 0u, static_cast<unsigned long long>(allocations),
        completed_count, committed_count, commit_order[0], commit_order[1],
        commit_order[2], commit_order[3]);
  }
  return ReadyOrder{
      .allocations = allocations,
      .ok = ok,
  };
}

} // namespace runtime_compute_host_detail
