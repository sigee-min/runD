#include "local.hpp"

namespace package_compute {

int Services() {
  using namespace graph_services;
  if (const int resources = CheckResourcePlan(); resources != 0) {
    return resources;
  }
  if (const int session = CheckSessionCompile(); session != 0) {
    return session;
  }
  return CheckExecution();
}

} // namespace package_compute
