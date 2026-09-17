# Installation and Release

The Policy Space runtime follows semantic versioning. Production evaluations should use an immutable Git tag, commit hash, or CI-built wheel. Editable installation is intended for local development only.

## Local development

```bash
git clone <repository-url> policy-space
cd policy-space
pip install -e ".[test]"
pytest -q
```

## Install an immutable version

Release tags use `v<major>.<minor>.<patch>`, for example:

```bash
pip install "policy-space @ git+ssh://<host>/<namespace>/policy-space.git@v0.3.0"
```

Replace the tag with a 40-character commit hash when the complete build input must be pinned. ManaEnv production dependencies use this form so later branch changes cannot affect an evaluation.

## Install a CI wheel

Each CI build publishes a `dist/` artifact. After downloading it:

```bash
pip install dist/policy_space-0.3.0-py3-none-any.whl
```

Clients that need only protocol and contract definitions can install the lightweight package from the same version:

```bash
pip install "policy-space-protocol @ git+ssh://<host>/<namespace>/policy-space.git@v0.3.0#subdirectory=packages/protocol"
```

## Release checks

Before a release:

```bash
pytest -q
python -m build
```

After tests and wheel installation checks pass, create a tag matching the version in `pyproject.toml`. Increase the major version for protocol-breaking changes, the minor version for compatible capabilities, and the patch version for compatible fixes.

## Export a public repository

The internal repository retains its development history and pinned commits. For an external release, export a verified tag into an empty repository so internal history is not published:

```bash
mkdir -p <empty-directory>
git archive --format=tar --prefix=policy-space/ v0.3.0 \
  | tar -xf - -C <empty-directory>
cd <empty-directory>/policy-space
git init
git add .
git commit -m "Initial public release: Policy Space v0.3.0"
```

Set the public remote and push the exported snapshot. Generate public snapshots from release tags, not development branches or internal repository mirrors.

[简体中文](release.zh-CN.md)
