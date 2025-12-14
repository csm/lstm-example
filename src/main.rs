use std::path::PathBuf;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use clap::{Parser, Subcommand};
use clap_verbosity_flag::Verbosity;
use log::{debug, info, Level};
use crate::model::{LSTMConfig, LSTMModel};
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
    command: Option<Commands>,

    #[command(flatten)]
    verbosity: Verbosity,
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
    },
    Predict {
        #[clap(long, value_name = "PATH")]
        file: PathBuf,

        #[clap(long, default_value = "./data")]
        checkpoint_dir: PathBuf,

        #[clap(long)]
        label: u16,
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
        Some(Commands::Predict { file, checkpoint_dir , label}) => {
            env_logger::builder()
                .filter_level(cli.verbosity.into())
                .init();
            info!("Predicting...");
            let model = LSTMModel::<MyBackend>::load_checkpoint(checkpoint_dir, &device)
                .expect("Failed to load checkpoint");
            debug!("Model loaded");
            model.infer(file, label, &device);
        }
        None => {
            println!("No command provided");
        }
    }
}
