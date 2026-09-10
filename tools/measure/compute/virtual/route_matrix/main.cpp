#include "internal.hpp"

#include <cstdio>
#include <string_view>

namespace {

void usage() {
  std::fputs(
      "usage: runD-compute-route-matrix <metal|vulkan> <core|full>\n"
      "       runD-compute-route-matrix --aggregate\n",
      stderr);
}

} // namespace

int main(const int argc, char **const argv) {
  using namespace rund::measure::compute::route_matrix;
  if (argc == 2 && std::string_view{argv[1]} == "--aggregate") {
    return aggregate() ? 0 : 1;
  }
  if (argc != 3) {
    usage();
    return 2;
  }
  ComputeBackend backend = ComputeBackend::Unavailable;
  ProfileMode profile = ProfileMode::Full;
  if (!parse_backend(argv[1], backend) || !parse_profile(argv[2], profile)) {
    usage();
    return 2;
  }
  print_header();
  return run_matrix(backend, profile) ? 0 : 1;
}
