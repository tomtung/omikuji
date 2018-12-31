use std::io::{stderr, Stderr};

pub(crate) type ProgressBar = pbr::ProgressBar<Stderr>;

pub(crate) fn create_progress_bar(total: u64) -> ProgressBar {
    ProgressBar::on(stderr(), total)
}
