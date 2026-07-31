#include "test/assert.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/channel.hpp>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

rund::task::Task<void> Send(rund::task::channel<std::uint32_t> *const pipe) {
  (void)co_await pipe->send(17u);
}

rund::task::Task<void> Receive(rund::task::channel<std::uint32_t> *const pipe,
                               std::atomic_bool *const received) {
  const auto value = co_await pipe->recv();
  if (!value || *value != 17u) {
    const std::string_view error = value.error();
    std::fprintf(stderr, "runtime.mixed-stress recv failed: %.*s\n",
                 static_cast<int>(error.size()), error.data());
  }
  received->store(value && *value == 17u, std::memory_order_release);
}

rund::task::Task<void> Readable(const rund::host::io::FdView ready_fd,
                                const int native,
                                std::atomic_bool *const ready) {
  const auto result = co_await rund::host::io::readable(ready_fd);
  char byte = 0;
  ready->store(result && ::read(native, &byte, 1u) == 1 && byte == 'r',
               std::memory_order_release);
}

rund::task::Task<void> Write(const int fd) {
  (void)co_await rund::task::yield();
  const char byte = 'r';
  if (::write(fd, &byte, 1u) != 1) {
    std::fprintf(stderr, "runtime.mixed-stress write failed: fd=%d errno=%d\n",
                 fd, errno);
  }
}

} // namespace

int RunRuntimeMixedStressContract() {
  constexpr std::size_t count = 1u << 16u;
  constexpr std::size_t jobs_per_backend = 4u;
  constexpr std::size_t scope_rounds = 16u;
  const unsigned hardware = std::thread::hardware_concurrency();
  const std::uint32_t workers = hardware == 0u ? 1u : hardware;
  std::vector<std::int32_t> input(count, 3);
  auto cpu_program =
      rund::compute::on(rund::compute::Target::cpu(workers))
          .map<std::int32_t>("mixed-stress-cpu", count,
                             [](auto value) { return value * 3 + 1; })
          .compile();
  auto gpu_program =
      rund::compute::on(rund::compute::Target::metal())
          .map<std::int32_t>("mixed-stress-gpu", count,
                             [](auto value) { return value * 5 + 2; })
          .compile();
  TEST_ASSERT(cpu_program);
  if (!gpu_program && gpu_program.code() == rund::compute::Code::Unavailable) {
    return 0;
  }
  TEST_ASSERT(gpu_program);
  std::vector<rund::compute::Job<std::int32_t(std::int32_t)>> cpu_jobs{};
  std::vector<rund::compute::Job<std::int32_t(std::int32_t)>> gpu_jobs{};
  cpu_jobs.reserve(jobs_per_backend);
  gpu_jobs.reserve(jobs_per_backend);
  for (std::size_t index = 0u; index < jobs_per_backend; ++index) {
    auto cpu_job = cpu_program->resident(input);
    auto gpu_job = gpu_program->resident(input);
    TEST_ASSERT(cpu_job);
    TEST_ASSERT(gpu_job);
    cpu_jobs.push_back(std::move(*cpu_job));
    gpu_jobs.push_back(std::move(*gpu_job));
  }

  rund::SessionConfig options{};
  options.id = 1u;
  options.workers = workers;
  options.scheduler.task_workers = workers;
  options.scheduler.task_capacity = 64u;
  options.scheduler.ready_queue_capacity = 64u;
  options.scheduler.channel_capacity = 2u;
  options.scheduler.channel_wait_capacity = 4u;
  rund::Session runtime{};
  TEST_ASSERT(runtime.open(options));

  std::vector<rund::compute::Submission> compute_tasks{};
  compute_tasks.reserve(jobs_per_backend * 2u);
  const auto scope = runtime.scope([&] {
    for (std::size_t index = 0u; index < jobs_per_backend; ++index) {
      compute_tasks.push_back(runtime.compute(cpu_jobs[index]).submit());
      compute_tasks.push_back(runtime.compute(gpu_jobs[index]).submit());
    }
    for (std::size_t round = 0u; round < scope_rounds; ++round) {
      std::atomic_bool received{false};
      std::atomic_bool io_ready{false};
      int pipe_fds[2]{-1, -1};
      TEST_ASSERT(::pipe(pipe_fds) == 0);
      rund::host::io::Fd ready_fd =
          rund::host::io::take_native_fd(::dup(pipe_fds[0]));
      TEST_ASSERT(ready_fd);
      auto pipe = rund::task::channel<std::uint32_t>::make(0u);
      const auto receiver =
          rund::task::spawn("mixed-receive", Receive(&pipe, &received));
      const auto sender = rund::task::spawn("mixed-send", Send(&pipe));
      const auto reader = rund::task::spawn(
          "mixed-readable", Readable(ready_fd.view(), pipe_fds[0], &io_ready));
      const auto writer = rund::task::spawn("mixed-writer", Write(pipe_fds[1]));
      TEST_ASSERT(rund::task::join(receiver, sender, reader, writer));
      const int read_close = ::close(pipe_fds[0]);
      if (read_close != 0) {
        std::fprintf(stderr, "runtime.mixed-stress read close failed: %d\n",
                     errno);
      }
      TEST_ASSERT(read_close == 0);
      const int write_close = ::close(pipe_fds[1]);
      if (write_close != 0) {
        std::fprintf(stderr, "runtime.mixed-stress write close failed: %d\n",
                     errno);
      }
      TEST_ASSERT(write_close == 0);
      if (!received.load(std::memory_order_acquire) ||
          !io_ready.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "runtime.mixed-stress round failed: %zu\n", round);
      }
      TEST_ASSERT(received.load(std::memory_order_acquire));
      TEST_ASSERT(io_ready.load(std::memory_order_acquire));
    }
  });
  if (!scope) {
    const std::string_view error = scope.error();
    std::fprintf(stderr, "runtime.mixed-stress scope failed: %.*s\n",
                 static_cast<int>(error.size()), error.data());
  }
  TEST_ASSERT(scope);
  TEST_ASSERT(compute_tasks.size() == jobs_per_backend * 2u);
  for (auto &task : compute_tasks) {
    TEST_ASSERT(task.wait());
  }
  for (std::size_t index = 0u; index < jobs_per_backend; ++index) {
    const auto cpu_output = cpu_jobs[index].read();
    const auto gpu_output = gpu_jobs[index].read();
    TEST_ASSERT(cpu_output);
    TEST_ASSERT(gpu_output);
    TEST_ASSERT((*cpu_output)[0] == 10);
    TEST_ASSERT((*gpu_output)[0] == 17);
  }
  TEST_ASSERT(runtime.close());
  return 0;
}
