from setuptools import setup


def build_native(spec):
    build = spec.add_external_build(
        cmd=["cargo", "build", "--release"], path="../c-api"
    )

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
    install_requires=["milksnake"],
    milksnake_tasks=[build_native],
)
