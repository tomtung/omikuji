from setuptools import setup
import traceback
import os


class OpenBlasConfig:
    NONE = 0
    STATIC = 1
    SYSTEM = 2


def make_milksnake_task(openblas_config):
    def build(spec):
        cmd = ["cargo", "build", "--release"]
        if openblas_config == OpenBlasConfig.NONE:
            pass
        elif openblas_config == OpenBlasConfig.STATIC:
            cmd.extend(["--features", "openblas-static"])
        elif openblas_config == OpenBlasConfig.SYSTEM:
            cmd.extend(["--features", "openblas"])
        else:
            assert False, "Unkown OpenBLAS Config: " + repr(openblas_config)

        build = spec.add_external_build(cmd=cmd, path="../c-api")

        spec.add_cffi_module(
            module_path="parabel._libparabel",
            dylib=lambda: build.find_dylib(
                "parabel", in_path="../c-api/target/release"
            ),
            header_filename=lambda: build.find_header(
                "parabel.h", in_path="../c-api/target/include"
            ),
            rtld_flags=["NOW", "NODELETE"],
        )

    return build


def show_message(*lines):
    try:
        import shutil

        columns = shutil.get_terminal_size((80, 20)).columns
    except ImportError:
        columns = 80

    print("=" * columns)
    for line in lines:
        print(line)

    print("=" * columns)


with open(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), os.path.pardir, "README.md"
    ),
    encoding="utf-8",
) as f:
    LONG_DESCRIPTION = "\n" + f.read()


def run_setup(with_openblas):
    setup(
        name="parabel",
        version="0.0.1",
        description=(
            "Python binding to Parabel-rs, a highly parallelized 🦀Rust implementation of Parabel "
            "for extreme multi-label classification"
        ),
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        python_requires=">=3.6.0",
        url="https://github.com/tomtung/parabel-rs",
        license="MIT",
        packages=["parabel"],
        zip_safe=False,
        platforms="any",
        setup_requires=["milksnake"],
        milksnake_tasks=[make_milksnake_task(with_openblas)],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Rust",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries",
        ],
    )


try:
    run_setup(OpenBlasConfig.SYSTEM)
except:
    traceback.print_exc()
    show_message("Failed to build with system OpenBLAS; try compiling bundled OpenBLAS")
    try:
        run_setup(OpenBlasConfig.STATIC)
    except:
        show_message("Failed to build with OpenBLAS; try building without OpenBLAS")
        run_setup(OpenBlasConfig.NONE)
        show_message(
            "Successfully built but without OpenBLAS; prediction speed will be affected"
        )
