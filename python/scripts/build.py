from cffi import FFI
from pathlib import Path

C_API_PATH = Path(__file__).absolute().parent.parent.parent / "c-api"
INCLUDE_PATH = C_API_PATH / "include"
HEADER_PATH = INCLUDE_PATH / "parabel.h"
OBJECT_PATH = C_API_PATH / "target" / "release" / "libparabel.a"

ffibuilder = FFI()
ffibuilder.set_source(
    "_libparabel",
    r"""
    #include "parabel.h"
    """,
    extra_objects=[str(OBJECT_PATH)],
    include_dirs=[str(INCLUDE_PATH)],
    extra_link_args=["-static"],
)
with open(HEADER_PATH) as f:
    ffibuilder.cdef(
        "\n".join([line for line in f if not line.startswith("#")]), override=True
    )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
