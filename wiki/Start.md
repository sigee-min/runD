# Quick Start

Install the published `1.0.0` Alpha, link one target, and run a typed Compute
Flow. The supported binary tuple is Darwin ARM64; Linux remains a validated
source candidate rather than a consumer release.

## 0. Verify And Install The SDK

Download these three adjacent files from the
[1.0.0 release](https://github.com/sigee-min/runD/releases/tag/1.0.0):

```text
rund-sdk-1.0.0-darwin-arm64.tar.gz
rund-sdk-1.0.0-darwin-arm64.sha256
rund-verify
```

Make the verifier executable, then let it validate and install the versioned
prefix:

```bash
chmod +x ./rund-verify
./rund-verify \
  ./rund-sdk-1.0.0-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.0-darwin-arm64.sha256 \
  "$PWD"
```

The destination must already exist and the versioned prefix must not. Before
publishing that prefix, the verifier checks the checksum, archive safety,
source and artifact identities, public CMake target, host architecture,
compiler, standard library, Apple SDK, and native backend dependencies.

A mismatch fails closed and leaves no installed prefix. Do not bypass a failed
check by extracting the archive manually; select the artifact whose sealed
tuple matches the consumer host.

## 1. Create One Flow

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(rund_first_flow LANGUAGES CXX)

find_package(runD 1.0.0 EXACT CONFIG REQUIRED)

add_executable(rund_first_flow main.cpp)
target_link_libraries(rund_first_flow PRIVATE runD::sdk)
```

`main.cpp`:

```cpp compile run source=package/tests/consumer/example/compute.cpp
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto value) { return value * 2 + 5; })
                    .collect();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
```

## 2. Configure, Build, Run

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/rund-sdk-1.0.0-darwin-arm64"
cmake --build build
./build/rund_first_flow
```

Do not point CMake at the archive, a copied `include/` directory, or a runD
source checkout. The verified prefix and `runD::sdk` are the public integration
boundary.

## 3. Run The Same Flow On A GPU

Backend choice is a value, not a second graph language:

```cpp fragment
rund::compute::Target::cpu()
rund::compute::Target::metal()
rund::compute::Target::vulkan()
```

Pass the chosen target to the same `rund::compute::on(target, input)` chain.
The full [three-backend parity example](https://github.com/sigee-min/runD#see-it) compares the
returned bytes directly. If a selected backend is unavailable, runD returns a
typed `Unsupported` or `Unavailable` result instead of falling back to CPU.

Continue with [Compute](https://github.com/sigee-min/runD/wiki/Compute) for graph composition and resident
execution, [GPU Performance](https://github.com/sigee-min/runD/wiki/Performance) for choosing an execution
shape, or [SDK Consumption](https://github.com/sigee-min/runD/wiki/SDK) for the complete package boundary.
