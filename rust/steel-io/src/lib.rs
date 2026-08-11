//! `.npy`/Parquet output and TOML runfile parsing for STEEL.

pub mod npy;
pub mod output;
pub mod runfile;

pub use output::{run_param_dir_name, write_figure3, write_run};
pub use runfile::RunFile;
