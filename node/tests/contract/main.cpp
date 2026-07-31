#include <rund/compute/backend.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "case/table.hpp"

namespace {

#if !defined(RUND_NODE_TEST_BACKEND_CPU)
#if defined(RUND_NODE_HAVE_METAL_SDK) && defined(RUND_NODE_HAVE_VULKAN_SDK)
constexpr std::array kComputeBackends{rund::compute::Backend::Cpu,
                                      rund::compute::Backend::Metal,
                                      rund::compute::Backend::Vulkan};
#elif defined(RUND_NODE_HAVE_METAL_SDK)
constexpr std::array kComputeBackends{rund::compute::Backend::Cpu,
                                      rund::compute::Backend::Metal};
#elif defined(RUND_NODE_HAVE_VULKAN_SDK)
constexpr std::array kComputeBackends{rund::compute::Backend::Cpu,
                                      rund::compute::Backend::Vulkan};
#else
constexpr std::array kComputeBackends{rund::compute::Backend::Cpu};
#endif
constexpr std::array<std::array<rund::compute::Backend, 2u>, 2u>
    kNativeBackends{{
        {rund::compute::Backend::Cpu, rund::compute::Backend::Metal},
        {rund::compute::Backend::Cpu, rund::compute::Backend::Vulkan},
    }};
#endif
constexpr std::array kCpuBackend{rund::compute::Backend::Cpu};

bool g_backend_selected = false;
std::size_t g_backend_index = 0u;
bool g_backend_selection_observed = false;

[[nodiscard]] bool SelectBackend(const char *const name) noexcept {
  constexpr std::array names{"cpu", "metal", "vulkan"};
  for (std::size_t index = 0u; index < names.size(); ++index) {
    if (std::strcmp(name, names[index]) == 0) {
      g_backend_selected = true;
      g_backend_index = index;
      return true;
    }
  }
  return false;
}

const rund::node::test_contract::Case *Find(const std::string_view name) {
  for (const auto &test : rund::node::test_contract::case_table()) {
    if (std::string_view{test.name} == name) {
      return &test;
    }
  }
  return nullptr;
}

int RunOne(const std::string_view name) {
  const rund::node::test_contract::Case *test = Find(name);
  if (test != nullptr) {
    g_backend_selection_observed = false;
    const int result = test->run();
    if (result == 0 && g_backend_selected && !g_backend_selection_observed) {
      std::fprintf(stderr,
                   "node contract case does not support backend selection: "
                   "%.*s\n",
                   static_cast<int>(name.size()), name.data());
      return 2;
    }
    return result;
  }
  std::fprintf(stderr, "unknown node contract case: %.*s\n",
               static_cast<int>(name.size()), name.data());
  std::fprintf(stderr, "available node contract cases:\n");
  for (const auto &test : rund::node::test_contract::case_table()) {
    const std::string_view case_name{test.name};
    std::fprintf(stderr, "  %.*s\n", static_cast<int>(case_name.size()),
                 case_name.data());
  }
  return 2;
}

int RunMany(const int count, const char *const *names) {
  for (int i = 0; i < count; ++i) {
    const std::string_view name{names[i]};
    std::printf("node.test: %.*s\n", static_cast<int>(name.size()),
                name.data());
    std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    const int rc = RunOne(name);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    std::printf("node.test.done: %.*s elapsed_us=%lld\n",
                static_cast<int>(name.size()), name.data(),
                static_cast<long long>(elapsed.count()));
    std::fflush(stdout);
    if (rc != 0) {
      std::fprintf(stderr, "node contract case failed: %.*s (%d)\n",
                   static_cast<int>(name.size()), name.data(), rc);
      return rc;
    }
  }
  return 0;
}

} // namespace

namespace rund::node::test_contract {

std::span<const rund::compute::Backend> selected_compute_backends() noexcept {
  g_backend_selection_observed = true;
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  return kCpuBackend;
#else
  if (!g_backend_selected) {
    return kComputeBackends;
  }
  if (g_backend_index == 0u) {
    return kCpuBackend;
  }
  return kNativeBackends[g_backend_index - 1u];
#endif
}

} // namespace rund::node::test_contract

int main(const int argc, const char **argv) {
  int case_option = 1;
  if (argc >= 3 && std::strcmp(argv[1], "--backend") == 0) {
    if (!SelectBackend(argv[2])) {
      std::fprintf(stderr, "unknown node contract backend: %s\n", argv[2]);
      std::fprintf(stderr,
                   "usage: %s [--backend cpu|metal|vulkan] "
                   "--case <name>...\n",
                   argv[0]);
      return 2;
    }
    case_option = 3;
  }
  if (argc >= case_option + 2 &&
      std::strcmp(argv[case_option], "--case") == 0) {
    return RunMany(argc - case_option - 1, argv + case_option + 1);
  }
  std::fprintf(stderr,
               "usage: %s [--backend cpu|metal|vulkan] "
               "--case <name>...\n",
               argv[0]);
  return 2;
}
