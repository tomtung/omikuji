import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def run(self):
        # Build Rust library first
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                "../c-api/Cargo.toml",
                "--release",
                "--verbose",
            ]
        )

        super().run()


setup(
    name="parabel",
    version="0.0.1",
    packages=find_packages(exclude=["scripts", "scripts.*"]),
    setup_requires=["cffi>=1.0.0"],
    install_requires=["cffi>=1.0.0"],
    ext_package="parabel",
    cmdclass={"build_ext": build_ext},
    cffi_modules=["scripts/build.py:ffibuilder"],
)
