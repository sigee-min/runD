#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

int main() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto value) { return value * 2 + 5; })
                    .collect();
  if (!output) {
    const std::string_view message = output.error();
    std::fprintf(stderr, "cpu failed (code=%u): %.*s\n",
                 static_cast<unsigned>(output.code()),
                 static_cast<int>(message.size()), message.data());
    return output.exit_code();
  }
  if (*output != std::vector<std::int32_t>{7, 9, 11, 13}) {
    std::fputs("unexpected output; expected [7, 9, 11, 13]\n", stderr);
    return 2;
  }
  std::printf("cpu: [%d, %d, %d, %d]\n", (*output)[0], (*output)[1],
              (*output)[2], (*output)[3]);
  return 0;
}
