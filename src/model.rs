use std::error::Error;
use std::path::PathBuf;
use burn::data::dataloader::batcher::Batcher;
use burn::nn::{BatchNorm, BatchNormConfig, BiLstm, BiLstmConfig, Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::{s, Backend, Config, Int, Module, ToElement};
use burn::record::{CompactRecorder, Recorder};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use log::info;
use crate::dataset::{UrbanSoundBatch, UrbanSoundBatcher, UrbanSoundItem};
use crate::train::{CNNTrainingConfig, TrainingConfig};

#[derive(Config, Debug)]
pub struct LSTMConfig {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_classes: usize,
    #[config(default = false)]
    bidirectional: bool,
}

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Option<Lstm<B>>,
    bilstm: Option<BiLstm<B>>,
    fc1: Linear<B>,
    relu: Relu,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct CNNConfig {
    input_height: usize,  // bands (frequency bins)
    input_width: usize,   // frames (time steps)
    num_classes: usize,
}

#[derive(Module, Debug)]
pub struct CNNModel<B: Backend> {
    conv2d_1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2d_2: Conv2d<B>,
    bn2: BatchNorm<B>,
    conv2d_3: Conv2d<B>,
    bn3: BatchNorm<B>,
    relu: Relu,
    max_pool_2d: MaxPool2d,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl CNNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CNNModel<B> {
        // Conv layers: 1 -> 32 -> 64 -> 128 channels
        // Input will be [batch, 1, height, width] = [batch, 1, 60, 85]
        let conv2d_1 = Conv2dConfig::new([1, 32], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv2d_2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv2d_3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        // Batch normalization layers
        let bn1 = BatchNormConfig::new(32).init(device);
        let bn2 = BatchNormConfig::new(64).init(device);
        let bn3 = BatchNormConfig::new(128).init(device);

        // MaxPool2d with 2x2 kernel and stride 2
        let max_pool_2d = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // After 2 pooling layers: height/4, width/4
        // [batch, 128, 15, 21] after global avg pool -> [batch, 128]
        let fc1 = LinearConfig::new(128, 64).init(device);
        let fc2 = LinearConfig::new(64, self.num_classes).init(device);

        let dropout = DropoutConfig::new(0.5).init();

        CNNModel {
            conv2d_1,
            bn1,
            conv2d_2,
            bn2,
            conv2d_3,
            bn3,
            relu: Relu::new(),
            max_pool_2d,
            dropout,
            fc1,
            fc2,
        }
    }
}

impl<B: Backend> CNNModel<B> {
    fn forward(&self, wav: Tensor<B, 3>) -> Tensor<B, 2> {
        // Input: [batch, seq_len, features] = [batch, 85, 60] (from LSTM format)
        // Need to convert to: [batch, channels, height, width] = [batch, 1, 60, 85]

        let dims = wav.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];      // 85 (time/width)
        let features = dims[2];      // 60 (frequency/height)

        // Reshape: [batch, seq_len, features] -> [batch, 1, features, seq_len]
        //          [batch, 85, 60] -> [batch, 1, 60, 85]
        let x = wav.swap_dims(1, 2)  // [batch, 60, 85]
            .reshape([batch_size, 1, features, seq_len]);

        // Normalize from [0, 255] to [0, 1]
        let x = x / 255.0;

        // Conv block 1: [batch, 1, 60, 85] -> [batch, 32, 60, 85]
        let x = self.conv2d_1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.max_pool_2d.forward(x);  // [batch, 32, 30, 42]

        // Conv block 2: [batch, 32, 30, 42] -> [batch, 64, 30, 42]
        let x = self.conv2d_2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);
        let x = self.max_pool_2d.forward(x);  // [batch, 64, 15, 21]

        // Conv block 3: [batch, 64, 15, 21] -> [batch, 128, 15, 21]
        let x = self.conv2d_3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);

        // Global Average Pooling: average over spatial dimensions [2, 3]
        let x = x.mean_dims(&[2, 3]);  // [batch, 128, 1, 1]
        let x = x.flatten(1, 3);  // [batch, 128] - flatten dims 1-3

        // Dropout
        let x = self.dropout.forward(x);

        // Classification head
        let x = self.relu.forward(self.fc1.forward(x));  // [batch, 64]
        let x = self.fc2.forward(x);  // [batch, num_classes]

        x
    }

    fn forward_classification(
        &self,
        wavs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>
    ) -> ClassificationOutput<B> {
        let output = self.forward(wavs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<UrbanSoundBatch<B>, ClassificationOutput<B>> for CNNModel<B> {
    fn step(&self, item: UrbanSoundBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item.sonograms, item.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<UrbanSoundBatch<B>, ClassificationOutput<B>> for CNNModel<B> {
    fn step(&self, item: UrbanSoundBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.sonograms, item.targets)
    }
}

impl<B: Backend> CNNModel<B> {
    pub fn load_checkpoint(checkpoint_dir: PathBuf, device: &B::Device) -> Result<CNNModel<B>, Box<dyn Error>> {
        info!("Loading CNN checkpoint from {:?}", checkpoint_dir.display());
        let config = CNNTrainingConfig::load(checkpoint_dir.join("config.json"))?;
        let record = CompactRecorder::new()
            .load(checkpoint_dir.join("model").into(), device)?;
        // Note: This assumes TrainingConfig has a cnn_model field
        // You may need to update TrainingConfig to support CNNConfig
        Ok(config.model.init(device).load_record(record))
    }

    pub fn infer(&self, file: PathBuf, label: u16, device: &B::Device) {
        let batcher = UrbanSoundBatcher {
            frames: 41,
            bands: 60,
            dataset: file.parent().unwrap().to_owned(),
        };
        let batch = batcher.batch(vec![UrbanSoundItem {
            slice_file_name: file.iter().last().unwrap().to_str().unwrap().to_string(),
            fs_id: "".to_string(),
            start: 0.0,
            end: 0.0,
            salience: 0,
            fold: "".to_string(),
            class_id: 0,
            class: "".to_string(),
            full_path: Some(file.clone()),
        }], device);
        let output = self.forward(batch.sonograms);
        // Output is [batch, num_classes] - already aggregated by the CNN
        let predicted_class = output.clone()
            .argmax(1)
            .into_scalar();

        println!("File: {}", file.display());
        println!("Predicted class: {} (expected: {})", class_id_to_class(predicted_class.to_u16()),
                 class_id_to_class(label));
        println!("Logits: {}", output.to_data());
    }
}

impl LSTMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LSTMModel<B> {
        if self.bidirectional {
            LSTMModel {
                lstm: None,
                bilstm: Some(BiLstmConfig::new(self.input_size, self.hidden_size, false).init(device)),
                fc1: LinearConfig::new(self.hidden_size * 2, self.hidden_size/2).init(device),
                relu: Relu::new(),
                fc2: LinearConfig::new(self.hidden_size/2, self.hidden_size/2).init(device),
                fc3: LinearConfig::new(self.hidden_size/2, self.num_classes).init(device),
            }
        } else {
            LSTMModel {
                lstm: Some(LstmConfig::new(self.input_size, self.hidden_size, false).init(device)),
                bilstm: None,
                fc1: LinearConfig::new(self.hidden_size, self.hidden_size/2).init(device),
                relu: Relu::new(),
                fc2: LinearConfig::new(self.hidden_size/2, self.hidden_size/2).init(device),
                fc3: LinearConfig::new(self.hidden_size/2, self.num_classes).init(device),
            }
        }
    }
}

impl<B: Backend> LSTMModel<B> {
    fn forward(&self, wav: Tensor<B, 3>) -> Tensor<B, 2> {
        let out = if let Some(ref lstm) = self.lstm {
            let (out, _) = lstm.forward(wav, None);
            out
        } else if let Some(ref bilstm) = self.bilstm {
            let (out, _) = bilstm.forward(wav, None);
            out
        } else {
            panic!("No LSTM or BiLSTM layer found");
        };

        // Get the last timestep: out is [batch, seq_len, hidden_size] or [batch, seq_len, hidden_size*2]
        let dims = out.dims();
        let seq_len = dims[1];
        // Select last timestep along dimension 1
        let out = out.narrow(1, seq_len - 1, 1).reshape([dims[0], dims[2]]);
        let out = self.relu.forward(self.fc1.forward(out));
        let out = self.relu.forward(self.fc2.forward(out));
        let out = self.fc3.forward(out);
        out
    }

    fn forward_classification(
        &self,
        wavs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>
    ) -> ClassificationOutput<B> {
        let output = self.forward(wavs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<UrbanSoundBatch<B>, ClassificationOutput<B>> for LSTMModel<B> {
    fn step(&self, item: UrbanSoundBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item.sonograms, item.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<UrbanSoundBatch<B>, ClassificationOutput<B>> for LSTMModel<B> {
    fn step(&self, item: UrbanSoundBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.sonograms, item.targets)
    }
}

fn class_id_to_class(class_id: u16) -> String {
    match class_id {
        0 => "Air Conditioner".to_string(),
        1 => "Car Horn".to_string(),
        2 => "Children Playing".to_string(),
        3 => "Dog Bark".to_string(),
        4 => "Drilling".to_string(),
        5 => "Engine Idling".to_string(),
        6 => "Gunshot".to_string(),
        7 => "Jackhammer".to_string(),
        8 => "Siren".to_string(),
        9 => "Street Music".to_string(),
        _ => "Unknown".to_string()
    }
}

impl<B: Backend> LSTMModel<B> {
    pub fn load_checkpoint(checkpoint_dir: PathBuf, device: &B::Device) -> Result<LSTMModel<B>, Box<dyn Error>> {
        info!("Loading checkpoint from {:?}", checkpoint_dir.display());
        let config = TrainingConfig::load(checkpoint_dir.join("config.json"))?;
        let record = CompactRecorder::new()
            .load(checkpoint_dir.join("model").into(), device)?;
        Ok(config.model.init(device).load_record(record))
    }

    pub fn infer(&self, file: PathBuf, label: u16, device: &B::Device) {
        let batcher = UrbanSoundBatcher {
            frames: 41,
            bands: 60,
            dataset: file.parent().unwrap().to_owned(),
        };
        let batch = batcher.batch(vec![UrbanSoundItem {
            slice_file_name: file.iter().last().unwrap().to_str().unwrap().to_string(),
            fs_id: "".to_string(),
            start: 0.0,
            end: 0.0,
            salience: 0,
            fold: "".to_string(),
            class_id: 0,
            class: "".to_string(),
            full_path: Some(file.clone()),
        }], device);
        let output = self.forward(batch.sonograms);
        // Output is [num_frames, num_classes] - average across frames to get single prediction
        let avg_logits = output.mean_dim(0); // Average across dimension 0 (frames) -> [10]

        // Reshape [10] to [1, 10], then argmax on dim 1 to get [1], then scalar
        let predicted_class = avg_logits.clone()
            .reshape([1, 10])
            .argmax(1)
            .into_scalar();

        println!("File: {}", file.display());
        println!("Predicted class: {} (expected: {})", class_id_to_class(predicted_class.to_u16()),
                 class_id_to_class(label));
        println!("Average logits: {}", avg_logits.to_data());
    }
}