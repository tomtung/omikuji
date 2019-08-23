from setuptools import setup


def build_native(spec):
    build = spec.add_external_build(cmd=["cargo", "build", "--release"], path="c-api")
    spec.add_cffi_module(
        module_path="omikuji._libomikuji",
        dylib=lambda: build.find_dylib("omikuji", in_path="target/release"),
        header_filename=lambda: build.find_header(
            "omikuji.h", in_path="target/include"
        ),
        rtld_flags=["NOW", "NODELETE"],
    )


setup(
    name="omikuji",
    version="0.1.2",
    author="Tom Dong",
    author_email="tom.tung.dyb@gmail.com",
    description=(
        "Python binding to Omikuji, an efficient implementation of Partioned Label Trees and its variations "
        "for extreme multi-label classification"
    ),
    python_requires=">=3.5",
    url="https://github.com/tomtung/omikuji",
    license="MIT",
    packages=["omikuji"],
    package_dir={"": "python-wrapper"},
    zip_safe=False,
    platforms="any",
    setup_requires=["milksnake"],
    install_requires=["milksnake"],
    milksnake_tasks=[build_native],
    milksnake_universal=False,
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
