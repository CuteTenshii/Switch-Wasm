# Contributing

Questions are welcome on the [Discord server](https://discord.gg/uhy2P4vyhE). Bugs and compatibility reports go in the [issue tracker](https://git.tenshii.moe/Tenshii/Switch-Wasm/issues).

## What can't go in

The emulator contains nothing of Nintendo's, and it has to stay that way. Keep all of the following out of the repository, issues, pull requests and logs you attach:

- Keys (`prod.keys`, `title.keys`), firmware, game files, save files and anything dumped from a console.
- Code from Nintendo's SDK, leaked or official, and anything derived from it.

Code adapted from another project is fine when its license allows it. Say where it came from in the pull request, and add the project to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) if it isn't there yet. Eden's code is GPL, which the project's license accepts; a project under a license the AGPL can't absorb can be read for reference but not copied from.

## License

Switch WASM is licensed under the [AGPL v3.0 or later](LICENSE). By opening a pull request you agree that your contribution is released under the same license.

## AI-assisted code

Using an AI assistant is fine. You are responsible for what it writes: read every line before you open the pull request, run it, and be able to explain it in review. Watch for code it reproduced from another project; that needs the same credit as code you adapted yourself.

## Finding your way around

The `docs/` folder has one file per subsystem (CPU, JIT, kernel, services, GPU, audio, video, the web frontend), and each one records why the code is the way it is. Read the one for the area you're changing first. [TODO.md](TODO.md) lists planned work and known gaps.

## Before opening a pull request

`bun install` installs a pre-commit hook that runs rustfmt and Clippy when Rust changed, the typecheck and Oxlint when TypeScript did, and Stylelint when CSS did. A pre-push hook runs the host and unit tests. Don't skip either of them.

The full set of checks:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
bun run typecheck
bun run lint
bun run stylelint
bun run test:unit
bun run test:e2e   # needs `make wasm` first
```

Host tests don't prove a change works in the browser. For anything the page or the WebAssembly build touches, build with `make wasm` and try it in a browser.

## Commits

- Use [Conventional Commits](https://www.conventionalcommits.org): `<type>(<scope>): <subject>`, for example `perf(gpu): skip the flush for copies that touch nothing held`. The subject is imperative and lowercase, with no trailing period.
- One change per commit. A refactor and the fix it enables are two commits.
- Use the body to explain why the change is needed.

## Compatibility reports

Report a game in the format of [docs/compat.md](docs/compat.md): the title, its title ID and version, the status (Fails, Boots, Menus, In-game or Playable), how far it gets, and the browser, OS and machine you ran it on. A screenshot and the log the page saves help. Don't attach anything from the list above.
