from pathlib import Path
import tempfile

from scripts.make_synthetic import make


def legacy_dataset() -> Path:
    global _fixture
    if _fixture is None:
        _fixture = tempfile.TemporaryDirectory(prefix="pnam-legacy-test-")
        make(Path(_fixture.name))
    return Path(_fixture.name)


_fixture = None
