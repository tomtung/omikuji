extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-w")
        .flag("-O3")
        .files(vec![
            "liblinear/blas/daxpy.c",
            "liblinear/blas/ddot.c",
            "liblinear/blas/dnrm2.c",
            "liblinear/blas/dscal.c",
            "liblinear/tron.cpp",
            "liblinear/linear.cpp",
        ])
        .compile("liblinear.a");
}
