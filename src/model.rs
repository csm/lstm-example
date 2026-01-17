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
use crate::train::TrainingConfig;

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
