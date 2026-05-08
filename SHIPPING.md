# Shipping paretobandits to GitHub + PyPI

Step-by-step instructions for publishing `paretobandits` to GitHub and PyPI from your Mac. Everything below is one-time setup the first time, plus a single `make release` per subsequent version.

The package itself is already build-verified: the wheel installs cleanly into a fresh venv, `twine check` passes, `pytest` shows 23/23 green, `ruff check` reports no errors. You're not debugging code — you're configuring publishing.

## TL;DR

```bash
# One-time setup
gh repo create apurvshukla/paretobandits --public --source=. --remote=origin --description "Multi-objective contextual bandits under distribution shift"
git add . && git commit -m "Initial release of paretobandits v0.3.0"
git push -u origin main

# Configure PyPI Trusted Publishing (browser, ~5 minutes — see below)

# Cut the release
make release VERSION=0.3.0
git push origin main && git push origin v0.3.0
```

That last `git push origin v0.3.0` triggers `.github/workflows/release.yml`, which builds, uploads to TestPyPI, then PyPI, then creates a GitHub release. No tokens, no manual `twine upload`.

## Detailed walkthrough

### 0. One-time cleanup before `git init`

There's a leftover `paretobandits-0.3.0/` directory in the project root from an in-place build that ran before output was redirected to `/tmp`. The sandbox couldn't delete it; do this from your Mac terminal first:

```bash
cd ~/Desktop/BanditControl/covar/paretobandits
rm -rf paretobandits-0.3.0/ build/ dist/ paretobandits.egg-info/
```

`.gitignore` already excludes `paretobandits-[0-9]*/` so future stray artifacts won't be committed even if you forget.

### 1. Initialize the git repo

From `~/Desktop/BanditControl/covar/paretobandits`:

```bash
cd ~/Desktop/BanditControl/covar/paretobandits
git init
git add .
git commit -m "Initial release of paretobandits v0.3.0

- Algorithm 1 (PCBShift) from Shukla & Kumar 2024
- Five baselines: SukKpotufe20, Turgay18, Auer16, ScalarizedUCB, RandomPlay
- Synthetic + Warfarin environments with five shift schedules
- Five evaluation metrics including the paper's d_p
- Higher-d contexts (d >= 1) supported by PCBShift and SyntheticShift
- 23/23 tests passing, ruff-clean"
```

Don't forget to set your git identity if you haven't already:
```bash
git config user.name "Apurv Shukla"
git config user.email "apurv.shukla@umich.edu"
```

### 2. Create the GitHub repo

If you have the GitHub CLI:
```bash
gh repo create apurvshukla/paretobandits \
    --public \
    --source=. \
    --remote=origin \
    --description "Multi-objective contextual bandits under distribution shift" \
    --homepage "https://github.com/apurvshukla/paretobandits"
git push -u origin main
```

Without `gh`: create the repo manually at https://github.com/new (name `paretobandits`, public, no README/license — we already have those), then:
```bash
git remote add origin https://github.com/apurvshukla/paretobandits.git
git branch -M main
git push -u origin main
```

After pushing, the `tests` workflow at `.github/workflows/test.yml` will run on Python 3.9–3.12 across Linux and macOS. Wait for it to go green before doing the release. If anything fails, fix it locally and force-push (this is still pre-release so rewriting history is fine).

### 3. Set up PyPI Trusted Publishing

This avoids storing API tokens in repo secrets. One-time setup, takes ~5 minutes.

**a) PyPI account.** If you don't have one: https://pypi.org/account/register/ — verify your email, enable 2FA (recommended).

**b) TestPyPI account.** Same drill at https://test.pypi.org/account/register/ — TestPyPI is a separate site with its own credentials. Treat the release pipeline's TestPyPI step as a smoke check.

**c) Reserve the project name.** Go to https://pypi.org/manage/account/publishing/ and add a "pending publisher":

| Field | Value |
|---|---|
| PyPI Project Name | `paretobandits` |
| Owner | `apurvshukla` |
| Repository name | `paretobandits` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Repeat the same form at https://test.pypi.org/manage/account/publishing/ but use environment name `testpypi`.

**d) Configure GitHub environments.** In your GitHub repo, go to **Settings → Environments** and create two environments named exactly `pypi` and `testpypi`. No secrets to add — the OIDC token is generated at workflow runtime. Optionally, add yourself as a required reviewer on the `pypi` environment so production uploads need a manual approval click.

### 4. Cut the release

```bash
cd ~/Desktop/BanditControl/covar/paretobandits
make release VERSION=0.3.0
```

This runs the recipe in `Makefile:release` which:
1. Updates `paretobandits/__init__.py:__version__` to `"0.3.0"`.
2. Commits the bump.
3. Creates the annotated tag `v0.3.0`.

Then push:

```bash
git push origin main
git push origin v0.3.0
```

The tag push triggers the release workflow. You can watch it at https://github.com/apurvshukla/paretobandits/actions. Sequence:

1. **build** — produces sdist + wheel, runs `twine check`, uploads as workflow artifact.
2. **publish-testpypi** — uploads to https://test.pypi.org/project/paretobandits/. Verify here first — check the project page, README rendering, and `pip install --index-url https://test.pypi.org/simple/ paretobandits` in a clean venv.
3. **publish-pypi** — uploads to https://pypi.org/project/paretobandits/. If you set up a required reviewer on the `pypi` environment, you'll need to click "Approve and deploy" in GitHub Actions for this to proceed.
4. **github-release** — creates a release on the repo, attaches the wheel + sdist, and auto-generates release notes from the commit log since the previous tag.

Total wall-clock time: usually 3–5 minutes.

### 5. Verify

```bash
pip install paretobandits --upgrade
python -c "import paretobandits; print(paretobandits.__version__)"     # → 0.3.0
```

Then run the quickstart from anywhere:
```bash
git clone https://github.com/apurvshukla/paretobandits.git /tmp/pb
cd /tmp/pb && python examples/quickstart.py
```

If both work, you've shipped.

## Subsequent releases

Editing the changelog and bumping the version is now a single command:

```bash
# 1. Update CHANGELOG.md by hand — add a new section under [Unreleased].
# 2. Bump version + tag + commit:
make release VERSION=0.4.0

# 3. Push:
git push origin main && git push origin v0.4.0
```

The release workflow does the rest.

## Troubleshooting

**`ruff check` fails in CI but passed locally.** Ruff version skew. Pin the version in `pyproject.toml`'s `[project.optional-dependencies] dev` block (e.g. `"ruff>=0.5,<0.6"`).

**TestPyPI upload says "version already exists".** Trusted publishing's `skip-existing: true` flag (already in `release.yml`) makes this a no-op. If it errors anyway, you've configured a non-trusted publisher; re-check the environment names match exactly.

**`twine check` fails with "long_description has syntax errors".** README markdown is rendered on PyPI; check for unbalanced backticks or broken links. Run `python -m twine check dist/*` locally to reproduce before pushing the tag.

**The Warfarin tests fail in CI.** CI doesn't ship the `iwpc_warfarin.xls` data file (see `.gitignore` philosophy — research data shouldn't live in package repos). The Warfarin module silently falls back to synthetic patient generation in that case; if a test asserts on real-data behavior, mark it with `@pytest.mark.skipif` on the file's existence.

**Forgot to update the changelog before tagging.** Either delete the tag (`git tag -d v0.3.0 && git push --delete origin v0.3.0`), update the changelog, and re-cut, or push a small follow-up release `v0.3.1` with just the changelog fix.

## Promoting the release

After v0.3.0 is live on PyPI, the highest-leverage things to do (in order):

1. **One adoption email.** DM Suk, Kpotufe, or someone in the active multi-objective bandit / DPO cluster, with a link to the GitHub repo and a one-paragraph pitch: "we built a benchmark library covering the closest baselines to our paper; would you consider using it for one experiment in your next work?" One external user before you submit the D&B paper is worth more than any extra README polish.
2. **Tweet/Bluesky.** Short post with the GitHub link, one screenshot of the quickstart ranking table. Tag relevant accounts.
3. **arXiv companion.** When you submit the algorithm paper to ICML/NeurIPS, point to the library in the abstract and the experiments section. Reviewers love reproducible artifacts.
4. **GitHub Topics.** On the repo page, add topics: `contextual-bandits`, `multi-objective-optimization`, `pareto`, `online-learning`, `reinforcement-learning`. Helps discoverability.
