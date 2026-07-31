# Release Checklist

Use this checklist before integrating a runD SDK artifact.

1. Confirm [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms) lists the tuple as
   `supported`. A producer or workflow candidate is not a consumer release.
2. Download the archive, `.sha256`, and `rund-verify` from the same release.
3. Run `sh ./rund-verify <archive> <checksum> [destination]`. Do not unpack the
   archive manually. Continue only when the command verifies the archive
   snapshot, entry safety, three seals, identity schemas, public-target hash,
   and exact host/compiler/standard-library/SDK/backend tuple, installs the
   versioned prefix, and prints its identity.
4. Confirm the installed SDK contains `lib/cmake/runD/runDConfig.cmake`.
5. Configure your project with `CMAKE_PREFIX_PATH` pointing at that prefix.
6. Use `find_package(runD 1.0.0 EXACT CONFIG REQUIRED)`.
7. Link `runD::sdk` `PRIVATE` when runD stays in implementation files, or
   `PUBLIC` when a documented runD declaration intentionally crosses the
   target's public C++ boundary.
8. Include only the direct entries owned by [API Stability](https://github.com/sigee-min/runD/blob/main/wiki/Stability);
   use [Public API Surface](https://github.com/sigee-min/runD/blob/main/wiki/Surface) only to navigate references.
9. Check [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms) and
   [API Stability](https://github.com/sigee-min/runD/blob/main/wiki/Stability) for the selected artifact.
