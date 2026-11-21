# Publishing checklist

This folder is the complete upload candidate for `https://github.com/kw78999/POP-PIM`. Generated environments,
builds, outputs, downloaded Liberty/CACTI files and traces are excluded by `.gitignore`.

## Still open (maintainer decisions)

- **License.** `LICENSE.md` records that no license has been chosen for the original POP-PIM code. Without one, the
  repository is visible but not open source. Add the chosen license text and copyright holder before announcing it.
- **Citation metadata.** Paper title, authors, affiliations and DOI/URL are not filled in. Complete `CITATION.md` and
  add a `CITATION.cff` once they are final.
- **CI.** `.github/workflows/ci.yml` has not run on GitHub yet. Its hardware job builds with GCC 10 and the Ubuntu
  22.04 Verilator (4.038, not the 4.028 used for the references) and runs the `quick` suite against the committed
  reference; check its first run.

## Upload

Create an empty repository on GitHub (no generated README or license), then from this folder:

```bash
python -m tools.check_release
git init
git add .
git status --short | head        # about 850 files, 10 MB; no outputs/, build/ or .venv/
git commit -m "POP-PIM 0.2.0: byte-frugal near-memory sparse attention (POP-PIM-S/Q)"
git branch -M main
git remote add origin https://github.com/kw78999/POP-PIM.git
git push -u origin main
```

Authenticate with SSH, a credential manager or a token prompt; never put credentials into a remote URL or a file.

## After changing the repository

Write new runs to fresh `outputs/` directories; do not edit committed reference records by hand. After an
intentional change to source or documentation, run the tests and `python -m tools.check_release`, review the diff,
then record the new contents with `python -m tools.check_release --write-manifest`. A changed simulator or input
requires regenerating every affected suite with `tools.reproduce` before its reference records are replaced.
