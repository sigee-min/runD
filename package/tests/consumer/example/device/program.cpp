#include "program/model.hpp"

using namespace package_device_program;

int main() {
  constexpr std::array targets{Target::cpu(2u), Target::metal(),
                               Target::vulkan()};
  Evidence cpu{};
  bool have_cpu = false;
  for (const Target target : targets) {
    const Backend requested = target.backend();
    auto opened = open(target);
    if (!opened) {
      if (requested == Backend::Cpu)
        return opened.exit_code();
      if (opened.error().empty() || (opened.code() != Code::Unsupported &&
                                     opened.code() != Code::Unavailable)) {
        return 2;
      }
      continue;
    }
    if (const int failure = CheckProfile(*opened, requested); failure != 0) {
      std::fprintf(stderr,
                   "installed pipeline failure profile backend=%u result=%d\n",
                   static_cast<unsigned>(requested), failure);
      return failure;
    }
    if (const int attribution = CheckAttribution(*opened, requested);
        attribution != 0) {
      std::fprintf(stderr,
                   "installed pipeline attribution backend=%u result=%d\n",
                   static_cast<unsigned>(requested), attribution);
      return attribution;
    }
    Evidence baseline{};
    Evidence observed{};
    const int baseline_result =
        RunTick(*opened, requested, PipelineProfile::None, baseline);
    if (baseline_result != 0) {
      std::fprintf(stderr, "installed pipeline baseline backend=%u result=%d\n",
                   static_cast<unsigned>(requested), baseline_result);
      return baseline_result;
    }
    const int profiled_result =
        RunTick(*opened, requested, PipelineProfile::Steps, observed);
    if (profiled_result != 0) {
      std::fprintf(stderr, "installed pipeline profiled backend=%u result=%d\n",
                   static_cast<unsigned>(requested), profiled_result);
      return profiled_result;
    }
    if (!same_execution_identity(baseline, observed)) {
      std::fprintf(stderr, "installed pipeline A/B identity backend=%u\n",
                   static_cast<unsigned>(requested));
      return 2;
    }
    if (requested == Backend::Cpu) {
      cpu = observed;
      have_cpu = true;
    } else if (!have_cpu || !same_evidence(cpu, observed)) {
      std::fprintf(stderr, "installed pipeline cross-backend backend=%u\n",
                   static_cast<unsigned>(requested));
      return 2;
    }
  }
  return have_cpu ? 0 : 2;
}
