#include "suite/model.hpp"

int main(const int argc, char **argv) {
  if (argc != 3 || !rund::measure::telemetry::Identity(argv[1]) || !rund::measure::telemetry::Identity(argv[2])) {
    std::fprintf(stderr, "usage: runD-telemetry-measure <manifest-sha256> "
                         "<artifact-sha256>\n");
    return 2;
  }
  try {
    return rund::measure::telemetry::Run(argv[1], argv[2]);
  } catch (const std::bad_alloc &) {
    (void)allocation::Stop();
    std::fprintf(stderr, "telemetry measurement allocation failed\n");
    return 1;
  } catch (...) {
    (void)allocation::Stop();
    std::fprintf(stderr, "telemetry measurement failed\n");
    return 1;
  }
}
