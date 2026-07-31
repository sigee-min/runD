#include "api.hpp"

namespace engine {

rund::Session::Result execute() {
  return rund::run(rund::SessionConfig{.workers = 1u}, [] {});
}

} // namespace engine
