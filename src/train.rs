use std::path::{Path, PathBuf};
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::prelude::{Config, Module};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{LearnerBuilder, LearningStrategy};
use burn::train::metric::{AccuracyMetric, LossMetric};
use log::{debug, info};
use crate::dataset::{UrbanSoundBatcher, UrbanSoundDataset};
use crate::model::LSTMConfig;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub dataset: PathBuf,
    pub checkpoint_dir: PathBuf,
    pub model: LSTMConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 10)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &PathBuf) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    dataset: PathBuf,
    checkpoint: PathBuf,
    frames: usize,
    bands: usize,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(&checkpoint);
    let checkpoint_dir = Path::new(&checkpoint).to_str().unwrap();
    config.save(format!("{checkpoint_dir}/config.json"))
        .expect("Failed to save config.json");

    B::seed(&device, config.seed);

    let batcher = UrbanSoundBatcher { frames, bands, dataset: dataset.clone() };

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(UrbanSoundDataset::new(&dataset).unwrap());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(UrbanSoundDataset::new(&dataset).unwrap());

    let model = config.model.init::<B>(&device);
    debug!("created model: {}", model);
    let learner = LearnerBuilder::new(checkpoint_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    result
        .model
        .save_file(format!("{checkpoint_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");

}