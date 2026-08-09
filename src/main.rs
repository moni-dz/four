use std::path::PathBuf;

use mimalloc::MiMalloc;

mod app;

#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;

fn main() {
    let initial_path = std::env::args_os().nth(1).map(PathBuf::from);
    app::run(initial_path.as_deref());
}
