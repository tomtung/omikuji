from setuptools import setup
import traceback


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


def run_setup(with_openblas):
    setup(
        name="parabel",
        version="0.0.1",
        packages=["parabel"],
        zip_safe=False,
        platforms="any",
        setup_requires=["milksnake"],
        milksnake_tasks=[make_milksnake_task(with_openblas)],
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
