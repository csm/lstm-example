use std::path::PathBuf;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use clap::{Parser, Subcommand};
use crate::model::LSTMConfig;
use crate::train::{train, TrainingConfig};

mod dataset;
mod model;
mod train;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train {
        #[clap(long, value_name = "PATH")]
        dataset: PathBuf,

        #[clap(long, default_value = "./data")]
        checkpoint_dir: PathBuf,

        #[clap(long, default_value = "41")]
        frames: usize,

        #[clap(long, default_value = "60")]
        bands: usize,

        #[clap(long, default_value = "2")]
        num_features: usize,

        #[clap(long, default_value = "10")]
        num_classes: usize,

        #[clap(long, default_value = "5300")]
        hidden_size: usize,

        #[clap(long, default_value = "1")]
        num_layers: usize,
    }
}

fn main() {
    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();
    match cli.command {
        Some(Commands::Train { dataset, checkpoint_dir, frames, bands, num_features, num_classes, hidden_size, num_layers}) => {
            train::<MyAutodiffBackend>(
                dataset.clone(), checkpoint_dir.clone(), frames, bands,
                TrainingConfig::new(dataset, checkpoint_dir,
                                    LSTMConfig::new(bands, hidden_size, num_layers, num_classes),
                                    AdamConfig::new(),
                ),
                device.clone(),
            )
        }
        None => {
            println!("No command provided");
        }
    }
}
