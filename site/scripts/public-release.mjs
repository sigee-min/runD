// The sole public-release identity. Keep this at the hosted artifact boundary;
// the CMake package version may be ahead while it is still a candidate.
const version = "1.0.4";
const artifactPrefix = `rund-sdk-${version}-darwin-arm64`;

export const publicRelease = Object.freeze({
  version,
  channel: "Alpha",
  tag: version,
  releaseUrl: `https://github.com/sigee-min/runD/releases/tag/${version}`,
  platform: "Darwin ARM64",
  support: "supported/release",
  // Frozen from package/docs/surface/headers.tsv at the published 1.0.4 tag.
  // The current source surface may add entries while 1.0.8 is a candidate.
  directHeaders: Object.freeze([
    "cluster/cluster.hpp",
    "math32/math32.hpp",
    "math64/math64.hpp",
    "rund/compute.hpp",
    "rund/compute/async.hpp",
    "rund/compute/math.hpp",
    "rund/compute/pipeline.hpp",
    "rund/compute/session.hpp",
    "rund/evidence.hpp",
    "rund/host.hpp",
    "rund/net.hpp",
    "rund/replay.hpp",
    "rund/rund.hpp",
    "rund/session.hpp",
    "rund/storage.hpp",
    "rund/task.hpp",
  ]),
  artifacts: Object.freeze({
    archive: `${artifactPrefix}.tar.gz`,
    checksum: `${artifactPrefix}.sha256`,
    verifier: "rund-verify",
    prefix: artifactPrefix,
  }),
});
