# Platform Support

Platform support is an artifact-and-evidence claim, not a statement that the
source happens to compile on a host. The canonical tuple matrix, status terms,
backend dependency rules, and determinism boundary are owned by the
[SDK platform support policy](https://github.com/sigee-min/runD/blob/main/package/docs/platform/support.md).

The public [runD 1.0.0 release](https://github.com/sigee-min/runD/releases/tag/1.0.0)
Darwin ARM64 archive, checksum, and verifier are classified as `supported`.
The Linux route produces a validated candidate only; a candidate is not a
consumer release until the canonical policy deliberately promotes its tuple
with matching release evidence.

The canonical registry binds status to artifact class: `validated` means
`candidate`, while `supported` means `release`. A `supported` tuple requires
an externally published archive, checksum, and self-tested `rund-verify`.
Neither a local candidate nor this wiki page can change support state.

Before choosing an archive:

1. Check the canonical policy for a `supported` tuple.
2. Run the adjacent `rund-verify`; it checks the archive's sealed identity,
   compiler, standard library, Apple SDK where applicable, and external backend
   dependencies against the current host before installing the prefix.
3. Follow the [Release Checklist](https://github.com/sigee-min/runD/wiki/Checklist).

This wiki page is a consumer navigation page. It intentionally carries no
second platform matrix or promotion authority.
