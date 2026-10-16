# Release checklist

1. **Version.** Set the new version in all five places (the C and Python tests check
   that the first three agree):
   - `CMakeLists.txt`: `project(potsynth VERSION x.y.z ...)`
   - `include/potsynth.h`: `POTSYNTH_VERSION_MAJOR`, `_MINOR`, `_PATCH`
   - `python/potsynth/__init__.py`: `__version__`
   - `doc/conf.py`: `release`
   - `CITATION.cff`: `version`

2. **Changelog.** In `CHANGELOG.md`, replace "(unreleased)" with the release date
   (YYYY-MM-DD). Add `date-released: YYYY-MM-DD` to `CITATION.cff`.

3. **Local checks.**
   - `tests/check.sh` (release, shared, generic, ASan+UBSan, TSan): all pass
   - `make -C doc html SPHINXOPTS="-W -n"`: no warnings
   - `python examples/example_mag.py` and `examples/example_grav.py` run

4. **CI.** Commit and push; the CI workflow must pass on every platform.

5. **Tag.** `git tag -a vx.y.z -m "potsynth x.y.z"` and `git push origin vx.y.z`. The
   tag starts the Wheels workflow, which builds and tests the wheels and the sdist.

6. **GitHub release.** Create a release from the tag, with the changelog section as the
   notes, and attach the wheels and the sdist (artifacts of the Wheels run).

7. **After the release.** Add a new "(unreleased)" section at the top of `CHANGELOG.md`.
