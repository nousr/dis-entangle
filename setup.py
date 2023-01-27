"""Setup script for dis_entangle."""

import os
from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath, encoding="utf-8") as f:
            return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="dis_entangle",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.1",
        license="MIT",
        description="A tool for generating masks for images.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="nousr",
        author_email="nousr@deforum.art",
        url="https://github.com/nousr/dis-entangle",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning", "txt2im", "datasets", "pytorch"],
        install_requires=REQUIREMENTS,
    )
