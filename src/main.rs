#![recursion_limit = "256"]

use std::path::PathBuf;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use clap::{Parser, Subcommand};
use clap_verbosity_flag::Verbosity;
use either::{Left, Right};
use log::{debug, info};
use crate::dataset::{UrbanSoundBatch, UrbanSoundBatcher, UrbanSoundDataset};
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

        #[clap(long, default_value = "512")]
        hidden_size: usize,

        #[clap(long, default_value = "1")]
        num_layers: usize,

        #[clap(long)]
        bidirectional: bool,
    },
    Predict {
        #[clap(long, value_name = "PATH")]
        file: PathBuf,

        #[clap(long, default_value = "./data")]
        checkpoint_dir: PathBuf,

        #[clap(long)]
        label: u16,
    },
    Precompute {
        #[clap(long, value_name = "PATH")]
        dataset: PathBuf,

        #[clap(long, default_value = "41")]
        frames: usize,

        #[clap(long, default_value = "60")]
        bands: usize,

        #[clap(long, default_value = "10")]
        batch_size: usize,
    }
}

fn precompute<B: Backend>(dataset: PathBuf, frames: usize, bands: usize, batch_size: usize, device: &B::Device) {
    let mem_dataset = UrbanSoundDataset::new(&dataset, bands).expect("Failed to read dataset metadata");
    let batcher = UrbanSoundBatcher {
        frames,
        bands,
        dataset,
    };
    let mut items = Vec::new();
    for i in 0..mem_dataset.dataset.len() {
        items.push(mem_dataset.dataset.get(i).unwrap());
        if items.len() >= batch_size {
            let _: UrbanSoundBatch<B> = batcher.batch(items.clone(), device);
            items.clear();
        }
    }
    if items.len() > 0 {
        let _: UrbanSoundBatch<B> = batcher.batch(items, device);
    }
}

fn main() {
    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();
    match cli.command {
        Some(Commands::Train { dataset, checkpoint_dir, frames, bands, num_features, num_classes, hidden_size, num_layers, bidirectional }) => {
            train::<MyAutodiffBackend>(
                dataset.clone(), checkpoint_dir.clone(), frames, bands,
                TrainingConfig::new(dataset, checkpoint_dir,
                                    LSTMConfig::new(bands, hidden_size, num_layers, num_classes).with_bidirectional(bidirectional),
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
        Some(Commands::Precompute { dataset, frames, bands, batch_size }) => {
            env_logger::builder()
                .filter_level(cli.verbosity.into())
                .init();
            precompute::<MyBackend>(dataset, frames, bands, batch_size, &device);
        }
        None => {
            println!("No command provided");
        }
    }
}
