use std::io::{Stderr, stderr};

pub(crate) type ProgressBar = pbr::ProgressBar<Stderr>;

pub(crate) fn create_progress_bar(total: u64) -> ProgressBar {
    ProgressBar::on(stderr(), total)
}
