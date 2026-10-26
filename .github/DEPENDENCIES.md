# Dependency maintenance

Dependabot checks Python (`uv`), the website (`bun`) and GitHub Actions every
Monday at 07:00 Europe/Rome. Ordinary minor and patch updates share one
multi-ecosystem pull request, with at most one routine PR open at a time.
Major version upgrades remain manual and do not
generate additional routine PRs. Security updates are grouped per ecosystem and
can arrive separately, including fixes that require a major upgrade.

The auto-merge workflow verifies the Dependabot commit signature and every
reported update type. Only minor and patch updates confined to dependency files
or existing workflow files qualify. Unknown update types and major security
fixes require manual review. The privileged workflow never checks out PR code.

## GitHub activation

The files must be published on the default branch before this configuration is
active. In the repository settings:

1. Enable the dependency graph, Dependabot alerts and Dependabot security updates
   under **Security / Advanced Security** (the section name can vary).
2. Under **General / Pull Requests**, enable **Allow auto-merge** and **Allow
   squash merging**.
3. Under **Branches**, create or update the classic branch protection rule for
   `main`. Require a pull request, require status checks, enable **Require branches
   to be up to date before merging**, and require these checks from GitHub Actions:
   - `quality (ubuntu-latest)`
   - `quality (windows-latest)`
   - `website`
4. Let the updated CI run once if the new `website` check is not yet selectable.
   Inspect **Insights / Dependency graph / Dependabot** to verify the single
   `dependencies` group and its update logs.

The workflow deliberately requires that classic protection rule, even if a
ruleset also protects the branch. It refuses to enable auto-merge when any of
the three required checks or the up-to-date requirement is missing. Existing
review requirements still apply: requiring a human approval also requires a
human approval for Dependabot PRs. The workflow does not approve its own PRs or
bypass repository rules.

For an eligible PR, GitHub merges only after the required checks and any reviews
pass. Failed checks leave the PR open. Dependabot may rebase it automatically;
each new head must pass CI. It cannot repair arbitrary application code or
provide a dependency fix before upstream releases one. Existing PRs are not
combined by this configuration.

## Website checks

CI installs the committed Bun lockfile, audits dependencies, checks types, runs
tests and lint, and builds both the regular site and GitHub Pages. It uses the
real website data from `giano-1.0.0-artifacts.zip` in release `v1.0.0`, verifies the
published SHA-256, and extracts only `artifacts/website/giano-history/`.
The extracted data is cached under the release checksum, and `prepare:data`
verifies the manifest's file hashes on every run. No inference or training runs
in this job. A cold cache downloads about 910 MB; the extracted history is about
2.2 GB. Release data, checkpoints and published site assets are not changed.

Configuration reference:
[multi-ecosystem updates](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/configuring-multi-ecosystem-updates),
[Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference),
[auto-merge](https://docs.github.com/en/code-security/tutorials/secure-your-dependencies/automate-dependabot-with-actions).
