use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, Relu};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::{s, Backend, Config, Int, Module};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use log::{debug, info};
use crate::dataset::UrbanSoundBatch;

#[derive(Config, Debug)]
pub struct LSTMConfig {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_classes: usize,
}

#[derive(Module, Debug)]
pub struct LSTMModel<B: Backend> {
    lstm: Lstm<B>,
    fc1: Linear<B>,
    relu: Relu,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl LSTMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LSTMModel<B> {
        LSTMModel {
            lstm: LstmConfig::new(self.input_size, self.hidden_size, false).init(device),
            fc1: LinearConfig::new(self.hidden_size, self.hidden_size/2).init(device),
            relu: Relu::new(),
            fc2: LinearConfig::new(self.hidden_size/2, self.hidden_size/2).init(device),
            fc3: LinearConfig::new(self.hidden_size/2, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> LSTMModel<B> {
    fn forward(&self, wav: Tensor<B, 3>) -> Tensor<B, 2> {
        let (out, _) = self.lstm.forward(wav, None);
        let dims = out.dims();
        let out = out.slice(s![0..dims[0], -1, 0..dims[2]]);
        let out = out.squeeze();
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