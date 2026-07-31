#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#if defined(__unix__) || defined(__APPLE__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <fcntl.h>

extern char **environ;
#endif

namespace rund::node::accel::detail {

bool RunProcess(const std::vector<std::string> &args) {
#if defined(__unix__) || defined(__APPLE__)
  if (args.empty()) {
    return false;
  }
  std::vector<char *> argv{};
  argv.reserve(args.size() + 1u);
  for (const std::string &arg : args) {
    argv.push_back(const_cast<char *>(arg.c_str()));
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions{};
  if (posix_spawn_file_actions_init(&actions) != 0) {
    return false;
  }
  if (posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                       O_WRONLY, 0) != 0 ||
      posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null",
                                       O_WRONLY, 0) != 0) {
    (void)posix_spawn_file_actions_destroy(&actions);
    return false;
  }
  pid_t pid = 0;
  const int spawn_result =
      posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), environ);
  (void)posix_spawn_file_actions_destroy(&actions);
  if (spawn_result != 0) {
    return false;
  }

  int status = 0;
  for (;;) {
    if (waitpid(pid, &status, 0) >= 0) {
      break;
    }
    if (errno != EINTR) {
      return false;
    }
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
  (void)args;
  return false;
#endif
}

} // namespace rund::node::accel::detail

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)
