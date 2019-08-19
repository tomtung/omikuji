from setuptools import setup
import os


def build_native(spec):
    build = spec.add_external_build(
        cmd=["cargo", "build", "--release"], path="../c-api"
    )
    spec.add_cffi_module(
        module_path="omikuji._libomikuji",
        dylib=lambda: build.find_dylib("omikuji", in_path="../c-api/target/release"),
        header_filename=lambda: build.find_header(
            "omikuji.h", in_path="../c-api/target/include"
        ),
        rtld_flags=["NOW", "NODELETE"],
    )


with open(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), os.path.pardir, "README.md"
    ),
    encoding="utf-8",
) as f:
    LONG_DESCRIPTION = "\n" + f.read()


setup(
    name="omikuji",
    version="0.1.1",
    description=(
        "Python binding to Omikuji, an efficient implementation of Partioned Label Trees and its variations "
        "for extreme multi-label classification"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    url="https://github.com/tomtung/omikuji",
    license="MIT",
    packages=["omikuji"],
    zip_safe=False,
    platforms="any",
    setup_requires=["milksnake"],
    install_requires=["milksnake"],
    milksnake_tasks=[build_native],
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
