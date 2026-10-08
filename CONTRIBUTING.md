# Contributing to Magma

Thanks for helping! Bug reports, new parser addons, kernels, docs and benchmarks on
other AMD GPUs are more than welcome!

## Commit messages: `Feat:` / `Fix:`

Every commit subject starts with exactly one of two prefixes, and CI rejects anything else.

| Prefix  | Use it when                                                                                     | Examples |
|---------|-------------------------------------------------------------------------------------------------|----------|
| `Feat:` | The commit **adds something new**: an element, a property, a parser addon, a kernel, a workflow, a docs page | `Feat: mgmosd draws class labels`<br>`Feat: InternImage parser addon` |
| `Fix:`  | **Anything else**: bug fixes, refactors, performance work, cleanups, formatting, docs or CI corrections | `Fix: cache DMABuf import to avoid per-frame re-imports`<br>`Fix: removed unnecessary syncs` |

Why: Fix commits are where regressions hide. Keeping them labelled makes it fast to walk
through everything that touched behaviour without adding a feature:

```sh
git log --oneline --grep '^Fix:'            # every non-feature change
git log --oneline --grep '^Fix:' -- gst-plugin/src/mgmpreproc.cpp
```

Not sure which prefix fits? If no user-visible capability was added, use `Fix:`.

**Only the prefix is enforced; the choice between them is checked loosely.** Deciding whether
a commit is a Feat or a Fix can be subjective, and commits sometimes mix both for convenience,
so nobody will block a PR over the classification. Just make an honest effort to pick the one
that best describes the commit.

**Exempt:** automated commits (bots, e.g. the badge updater) and "code movement" such as
merges and reverts don't need a prefix.

## Building and testing

You need ROCm 7.2.x, GStreamer >= 1.19, meson and ninja. The Docker image
(`docker compose up magma`) is the easiest way to get them.

```sh
meson setup build --buildtype=debug
meson compile -C build
meson test -C build --print-errorlogs             # everything (needs an AMD GPU)
meson test -C build --no-suite gpu --print-errorlogs  # what CI runs
```

Tests that need a GPU belong to the `gpu` suite (`suite: 'gpu'` in
`gst-plugin/tests/meson.build`). Hosted CI has no GPU, so please run that suite
locally if you touched kernels, pre-processing or inference.

## Code style

- C/C++/HIP is formatted with **clang-format 22.1.8** using the repo's `.clang-format`
  (LLVM based, tabs for indentation, 200 columns). CI runs `clang-format --dry-run --Werror`.
  Format before committing:
  ```sh
  git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.hip' | xargs clang-format -i
  ```
- GStreamer/GObject conventions: `G_DEFINE_TYPE`, `gst_` prefixes, `PROP_*` enums.
- Build system is Meson only (no CMake).

## License headers

Magma is LGPL-3.0-or-later and follows the [REUSE](https://reuse.software) spec. New
source files start with:

```cpp
// SPDX-FileCopyrightText: <year> <your name or handle>
// SPDX-License-Identifier: LGPL-3.0-or-later
```

Third-party files keep their original notice and get an entry in `REUSE.toml`.
`python3 scripts/check-license.py` (needs `reuse`) runs the same check as CI, including
whether copyright years are up to date.

By contributing you agree your work is released under the project's license.

## Using AI tools

Allowed, as a tool rather than an autopilot: you must understand, review and be able
to defend every line you submit. Core code (kernels, memory handling, pipeline logic)
should be read and checked by hand. Don't commit agent artifacts (`CLAUDE.md`,
`AGENTS.md`, `.claude/`, graph dumps, etc.).

## Pull requests

- Branch off `dev`; PRs target `dev`. `master` only receives merges from `dev`.
- Keep PRs focused. CI (build, CPU tests, format, license) must be green.
- For performance changes, include before/after numbers and the GPU you measured on.

## Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
