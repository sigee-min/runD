# Contributing to runD

Thank you for contributing.

The repository documentation is the source of truth for architecture,
behavior, naming, and verification. Start with [`docs/README.md`](docs/README.md)
and then read the nearest subsystem documentation for the code you intend to
change.

## Development workflow

1. Keep each behavior under one documented owner.
2. Update documentation, implementation, and contract tests together.
3. Run the narrowest focused test that proves the change.
4. Widen verification when the change crosses a shared boundary.
5. Keep generated files and local evidence under `.cache/`.

Useful commands:

```sh
tools/test/run --list
tools/test/run <case>
tools/check/run
tools/release/run
```

The exact workflow and evidence rules are documented in
[`docs/process/workflow.md`](docs/process/workflow.md) and
[`docs/process/evidence.md`](docs/process/evidence.md).

## Pull requests

Describe the contract being changed, the owning documentation and source
paths, and the commands used to verify it. Do not include generated build
outputs, caches, credentials, or unrelated changes.

By contributing, you agree that your contribution is licensed under the
repository's [MIT License](LICENSE).
