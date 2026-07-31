#include "local.hpp"

#include <string>

#include <unistd.h>

namespace runtime_task_host_io {

Fd::~Fd() {
  if (owned && native >= 0) {
    static_cast<void>(::close(native));
  }
}

Temp::~Temp() {
  if (!path.empty()) {
    static_cast<void>(::unlink(path.c_str()));
  }
}

std::string Path(const char *const suffix) {
  return std::string{".cache/hostio-contract-"} +
         std::to_string(static_cast<long long>(::getpid())) + "-" + suffix;
}

} // namespace runtime_task_host_io
