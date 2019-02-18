from setuptools import setup
import sys

# Add option to enable OpenBLAS; a bit hacky but works
if "--with-openblas" in sys.argv:
    with_openblas = True
    sys.argv.remove("--with-openblas")
else:
    with_openblas = False


def build_native(spec):
    cmd = ["cargo", "build", "--release"]
    if with_openblas:
        cmd.extend(["--features", "openblas"])

    build = spec.add_external_build(cmd=cmd, path="../c-api")

    spec.add_cffi_module(
        module_path="parabel._libparabel",
        dylib=lambda: build.find_dylib("parabel", in_path="../c-api/target/release"),
        header_filename=lambda: build.find_header(
            "parabel.h", in_path="../c-api/target/include"
        ),
        rtld_flags=["NOW", "NODELETE"],
    )


setup(
    name="parabel",
    version="0.0.1",
    packages=["parabel"],
    zip_safe=False,
    playforms="any",
    setup_requires=["milksnake"],
    milksnake_tasks=[build_native],
)
