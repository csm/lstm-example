use std::error::Error;
use std::fs::{create_dir, File};
use std::io::Write;
use std::path::PathBuf;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::prelude::{Backend, ElementConversion, Int, TensorData};
use burn::Tensor;
use burn::tensor::DType;
use fon::Audio;
use fon::chan::{Ch16, Ch32, Channel};
use gif::{ColorOutput, DecodeOptions, Decoder, Encoder, Frame};
use hound::{SampleFormat, WavReader, WavSpec};
use serde::Deserialize;
use sonogram::{ColourGradient, ColourTheme, FrequencyScale, SpecOptionsBuilder};
use log::{debug, info, warn};
use mel_spec::mel::interleave_frames;
use mel_spec::prelude::{MelSpectrogram, Spectrogram};
use mel_spec::quant::{tga_8bit, load_tga_8bit};
use resampler::{ResamplerFft, SampleRate};
use rubato::FftFixedIn;

#[derive(Deserialize, Debug, Clone)]
pub struct UrbanSoundItem {
    pub slice_file_name: String,
    #[serde(rename = "fsID")]
    pub fs_id: String,
    pub start: f64,
    pub end: f64,
    pub salience: i16,
    pub fold: String,
    #[serde(rename = "classID")]
    pub class_id: i16,
    pub class: String,

    #[serde(skip)]
    pub full_path: Option<PathBuf>,
}

pub struct UrbanSoundDataset {
    pub path: PathBuf,
    pub dataset: InMemDataset<UrbanSoundItem>,
    pub bands: usize,
}

impl UrbanSoundDataset {
    pub fn new(path: &PathBuf, bands: usize) -> Result<Self, std::io::Error> {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');
        let csv_path = path.join("metadata").join("UrbanSound8K.csv");
        let dataset = InMemDataset::from_csv(csv_path, rdr)?;
        let dataset = UrbanSoundDataset { path: path.clone(), dataset, bands };
        Ok(dataset)
    }
}

impl Dataset<UrbanSoundItem> for UrbanSoundDataset {
    fn get(&self, index: usize) -> Option<UrbanSoundItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone, Default)]
pub struct UrbanSoundBatcher {
    pub frames: usize,
    pub bands: usize,
    pub dataset: PathBuf,
}

#[derive(Clone, Debug)]
pub struct UrbanSoundBatch<B: Backend> {
    pub sonograms: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

fn wav_to_specs(bands: usize, wav_path: &PathBuf) -> Result<(WavSpec, Vec<Vec<u8>>), Box<dyn Error>> {
    let handle = File::open(wav_path)
        .map_err(|e| format!("Failed to open {:?}: {}", wav_path, e))?;
    let mut wav = WavReader::new(handle)
        .map_err(|e| format!("Failed to read WAV {:?}: {}", wav_path, e))?;
    let spec = wav.spec();
    info!("read wav {:?}: channels={}, sample_rate={}, bits={}, format={:?}",
           wav_path.file_name().unwrap_or_default(), spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format);

    let mels_path = wav_path.with_extension("mel");
    if mels_path.exists() && mels_path.is_dir() {
        // load precomputed mels from tga chunks.
        let mut entries: Vec<(usize, PathBuf)> = std::fs::read_dir(&mels_path)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let file_name = path.file_name()?.to_str()?;
                if file_name.starts_with("chunk_") && file_name.ends_with(".tga") {
                    let num_str = file_name.strip_prefix("chunk_")?.strip_suffix(".tga")?;
                    let num: usize = num_str.parse().ok()?;
                    Some((num, path))
                } else {
                    None
                }
            })
            .collect();
        entries.sort_by_key(|(num, _)| *num);

        let tgas: Result<Vec<Vec<u8>>, _> = entries
            .into_iter()
            .map(|(_, path)| {
                let path_str = path.to_str().ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid path encoding")
                })?;
                let floats = load_tga_8bit(path_str)?;
                // Convert f32 (0.0-1.0 range) to u8 (0-255 range)
                Ok(floats.into_iter().map(|f| (f * 255.0) as u8).collect())
            })
            .collect();
        info!("returning cached tga for {}", mels_path.display());
        return match tgas.map_err(|e: std::io::Error| e.into()) {
            Ok(tgas) => Ok((spec, tgas)),
            Err(e) => Err(e),
        }
    }

    // Parameterize these?
    let fft_size = 400;
    let hop_size = 100;
    let sample_rate = spec.sample_rate as f64;

    let mut stft = Spectrogram::new(fft_size, hop_size);
    let mut mel = MelSpectrogram::new(fft_size, sample_rate, bands);

    let samples: Box<dyn Iterator<Item=Result<f32, _>>> = match spec.sample_format {
        SampleFormat::Float => Box::new(wav.samples::<f32>()),
        SampleFormat::Int => Box::new(wav.samples::<i32>().map(|i| i.map(|i| i as f32))),
    };

    let mut mel_frames: Vec<ndarray::Array2<f64>> = Vec::new();
    let samples: Result<Vec<f32>, _> = samples.collect();

    for batch in samples?.windows(hop_size) {
        if let Some(fft_frame) = stft.add(&batch) {
            let mel_frame = mel.add(&fft_frame);
            mel_frames.push(mel_frame);
        }
    }
    create_dir(&mels_path)?;
    let interleaved = interleave_frames(mel_frames.as_slice(), false, 100);
    let tgas = tga_8bit(interleaved.as_slice(), bands);
    for (i, tga) in tgas.iter().enumerate() {
        let path = mels_path.join(format!("chunk_{}.tga", i));
        let mut file = File::create(path)?;
        file.write_all(tga.as_slice())?;
        debug!("wrote tga {}/chunk_{}.tga", mels_path.display(), i);
    }
    Ok((spec, tgas))
}

impl<B: Backend> Batcher<B, UrbanSoundItem, UrbanSoundBatch<B>> for UrbanSoundBatcher {
    fn batch(&self, items: Vec<UrbanSoundItem>, device: &B::Device) -> UrbanSoundBatch<B> {
        let mut wavs: Vec<Tensor<B, 3>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1, Int>> = Vec::new();
        for item in items {
            let wav_path = item.full_path;
            let wav_path = wav_path.or(Some(self.dataset.join("audio").join(format!("fold{}", item.fold)).join(&item.slice_file_name))).unwrap();
            let specs = wav_to_specs(self.bands, &wav_path);
            match specs {
                Ok((wav_spec, specs)) => {
                    for spec in specs {
                        let width = spec.len() / self.bands;

                        // Grayscale spectrogram: [height, width] = [bands, natural_width]
                        let tensor_2d = Tensor::<B, 2>::from_data(
                            TensorData::from_bytes_vec(spec, [self.bands, width], DType::U8).convert::<f32>(),
                            device
                        );
                        // Transpose to [width, height] = [natural_width, bands] for LSTM (time, freq)
                        let transposed = tensor_2d.swap_dims(0, 1);
                        // Reshape to [1, seq_len, features] = [1, natural_width, bands] for LSTM input
                        let tensor_3d = transposed.reshape([1, width, self.bands]);
                        wavs.push(tensor_3d);
                        targets.push(Tensor::<B, 1, Int>::from_data([(item.class_id as i64).elem::<B::IntElem>()], device));
                    }
                }
                Err(e) => {
                    warn!("Error loading audio specs for {:?}: {}", wav_path.display(), e);
                    // Continue with next file instead of failing the whole batch
                }
            }
        }
        debug!("num wavs: {} wavs: {:?}, num targets: {} targets: {:?}", wavs.len(), wavs, targets.len(), targets);

        let sonograms = Tensor::cat(wavs, 0);
        let targets = Tensor::cat(targets, 0);
        debug!("wavs shape: {:?}, targets shape: {:?}", sonograms.shape(), targets.shape());
        UrbanSoundBatch { sonograms, targets }
    }
}