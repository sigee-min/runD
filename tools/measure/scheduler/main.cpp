#include "suite/model.hpp"

namespace {

[[nodiscard]] bool ParseCount(const char *const text,
                              std::uint32_t &value) noexcept {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  value = static_cast<std::uint32_t>(parsed);
  return true;
}

} // namespace



int main(const int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "latency") {
    return rund::measure::scheduler::Latency();
  }
  std::uint32_t count = 0u;
  if ((argc == 3 || argc == 4) && std::string_view{argv[1]} == "memory" &&
      ParseCount(argv[2], count)) {
    std::uint32_t task_workers = 1u;
    if (argc == 4) {
      if (std::string_view{argv[3]} == "host") {
        task_workers = std::thread::hardware_concurrency();
        if (task_workers == 0u) {
          task_workers = 1u;
        }
      } else if (!ParseCount(argv[3], task_workers) || task_workers == 0u) {
        std::fprintf(stderr, "memory worker count must be positive or host\n");
        return 2;
      }
    }
    return rund::measure::scheduler::Memory(count, task_workers);
  }
  if ((argc == 3 || argc == 4 || argc == 5) &&
      std::string_view{argv[1]} == "scale" && ParseCount(argv[2], count)) {
    std::uint32_t task_workers = 1u;
    if (argc >= 4) {
      if (std::string_view{argv[3]} == "host") {
        task_workers = std::thread::hardware_concurrency();
        if (task_workers == 0u) {
          task_workers = 1u;
        }
      } else if (!ParseCount(argv[3], task_workers) || task_workers == 0u) {
        std::fprintf(stderr, "scale worker count must be positive or host\n");
        return 2;
      }
    }
    std::uint32_t payload_ops = 0u;
    if (argc == 5 && !ParseCount(argv[4], payload_ops)) {
      std::fprintf(stderr, "scale payload ops must be nonnegative\n");
      return 2;
    }
    return rund::measure::scheduler::Scale(count, task_workers, payload_ops);
  }
  std::fprintf(stderr, "usage: runD-scheduler-measure latency|"
                       "scale <task-count> [workers|host] [payload-ops]|"
                       "memory <task-count> [workers|host]\n");
  return 2;
}
