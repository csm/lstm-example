use std::error::Error;
use std::fs::File;
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
use mel_spec::quant::tga_8bit;
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

struct UrbanSoundSpecItem {
    pub file_name: String,
    pub spec: Vec<ndarray::Array2<f64>>,
    pub class_id: i16,
}

impl Dataset<UrbanSoundItem> for UrbanSoundDataset {
    fn get(&self, index: usize) -> Option<UrbanSoundItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl Dataset<UrbanSoundSpecItem> for UrbanSoundDataset {
    fn get(&self, index: usize) -> Option<UrbanSoundSpecItem> {
        if let Some(item) = self.dataset.get(index) {
            let path = if let Some(p) = item.full_path {
                p.clone()
            } else {
                let mut full_path = self.path.clone();
                full_path.push(item.slice_file_name);
                full_path
            };
            let spec = wav_to_specs(self.bands, &path);
            match spec {
                Ok(spec) => Some(
                    UrbanSoundSpecItem {
                        file_name: "".to_string(),
                        spec,
                        class_id: item.class_id,
                    }
                ),
                Err(e) => {
                    info!("Error loading spec from {:?}: {:?}", path, e);
                    None
                }
            }
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// ============================================================================
// Sequenced Dataset for Stateful LSTM
// ============================================================================

/// Represents a sequence of windows from a single audio file
/// Used for stateful LSTM processing where LSTM state is maintained across windows
#[derive(Clone, Debug)]
pub struct SequencedUrbanSoundItem {
    /// Audio file identifier
    pub file_name: String,
    // Mel spectrogram
    pub specs: Vec<ndarray::Array2<f64>>,
    /// Class label for this audio file
    pub class_id: i16,
}

/// Dataset that groups windows by audio file for sequential LSTM processing
#[derive(Clone)]
pub struct UrbanSoundSequenceDataset {
    pub path: PathBuf,
    pub items: Vec<SequencedUrbanSoundItem>,
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

fn read_gif_file(path: &PathBuf) -> Result<Vec<u8>, std::io::Error> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let frame = decoder.read_next_frame()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    match frame {
        Some(f) => Ok(f.buffer.clone().into()),
        None => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "")),
    }
}

fn wav_to_specs(bands: usize, wav_path: &PathBuf) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
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

    for sample in samples {
        let sample = sample?;
        let buf = [sample];
        if let Some(fft_frame) = stft.add(&buf) {
            let mel_frame = mel.add(&fft_frame);
            mel_frames.push(mel_frame);
        }
    }
    let interleaved = interleave_frames(mel_frames.as_slice(), false, 100);
    let tgas = tga_8bit(interleaved.as_slice(), bands);
    for (i, tga) in tgas.iter().enumerate() {
        let mut path = mels_path.clone();
        path.push(format!("chunk_{}.tga", i));
        let mut file = File::create(path)?;
        file.write_all(tga.as_slice())?;
    }
    Ok(tgas)
}

fn sample_to_spectrogram(
    bands: usize,
    wav_path: &PathBuf,
    out_spec: WavSpec,
    mut gradient: &mut ColourGradient,
    mut offset: usize,
    window_samples: &[i16]) -> Result<(Vec<u8>, usize, usize), Box<dyn Error>> {
    let mut spec_builder = SpecOptionsBuilder::new(2048)
        .load_data_from_memory(window_samples.to_vec(), out_spec.sample_rate)
        .build()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build spectrogram at offset {}: {:?}", offset, e)))?;

    debug!("computing spectrogram...");
    let mut spectrogram = spec_builder.compute();

    // Calculate natural width based on audio duration and FFT parameters
    // For 2 seconds at 44100 Hz with FFT 2048 and hop 1024:
    // frames = (samples - fft_size) / hop_size + 1 ≈ (88200 - 2048) / 1024 + 1 ≈ 85
    let fft_size = 2048;
    let hop_size = fft_size / 2;
    let natural_width = ((window_samples.len() - fft_size) / hop_size + 1).max(1);
    let height = bands;

    info!("converting to rgba with natural_width={}, height={}...", natural_width, height);
    // Get RGBA bytes for this spectrogram
    let rgba_bytes = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        spectrogram.to_rgba_in_memory(
            FrequencyScale::Log,
            &mut gradient,
            natural_width,
            height,
        )
    })) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(format!(
                "TooWide or other error converting spectrogram for {:?} with width={}, height={}",
                wav_path, natural_width, height
            ).into());
        }
    };

    debug!("successfully created spectrogram, rgba_bytes len: {}", rgba_bytes.len());

    // Convert RGBA to grayscale using luminosity method: 0.299*R + 0.587*G + 0.114*B
    let grayscale_bytes: Vec<u8> = rgba_bytes
        .chunks_exact(4)
        .map(|rgba| {
            let r = rgba[0] as f32;
            let g = rgba[1] as f32;
            let b = rgba[2] as f32;
            // Luminosity method
            (0.299 * r + 0.587 * g + 0.114 * b) as u8
        })
        .collect();
    Ok((grayscale_bytes, natural_width, height))
}

fn to_u8(v: Vec<i16>) -> Vec<u8> {
    v.iter().flat_map(|x| vec![(x >> 8) as u8, (x & 0xff) as u8]).collect()
}

fn to_f32(v: Vec<i16>) -> Vec<f32> {
    v.chunks_exact(2).map(|v| f32::from_bits(((v[0] as u32) << 16) | (v[1] as u32))).collect()
}

fn to_f64(v: Vec<i16>) -> Vec<f64> {
    v.chunks_exact(4).map(|v| f64::from_bits(
        ((v[0] as u64) << 48) |
            ((v[1] as u64) << 32) |
            ((v[2] as u64) << 16) |
            (v[3] as u64)
    )).collect()
}

fn resample_wav<Chan: Channel, const CH: usize, B>(
    spec: WavSpec, wav: &mut WavReader<File>
) -> Result<Vec<i16>, Box<dyn Error>>
    where B: Into<Box<[fon::Frame<Chan, CH>]>> {
    let channels = spec.channels as usize;
    let in_hz = hz_to_sample_rate(spec)?;
    let mut resampler = FftFixedIn::new(spec.sample_rate as usize,
                                        41_000, 1024, 1, channels)?;

}

fn hz_to_sample_rate(spec: WavSpec) -> Result<SampleRate, Box<dyn Error>> {
    match spec.sample_rate {
        16_000 => Ok(SampleRate::Hz16000),
        22_050 => Ok(SampleRate::Hz22050),
        32_000 => Ok(SampleRate::Hz32000),
        44_100 => Ok(SampleRate::Hz44100),
        48_000 => Ok(SampleRate::Hz48000),
        88_200 => Ok(SampleRate::Hz88200),
        96_000 => Ok(SampleRate::Hz96000),
        176_400 => Ok(SampleRate::Hz176400),
        192_000 => Ok(SampleRate::Hz192000),
        384_000 => Ok(SampleRate::Hz384000),
        _ => Err(format!("Unsupported sample rate: {}", spec.sample_rate).into())
    }
}

// This function was AI-generated; it should be reviewed.
fn convent_to_mono(spec: WavSpec, out_spec: WavSpec, samples: Vec<i16>) -> Vec<i16> {
    match (spec.channels == 2, spec.sample_rate != out_spec.sample_rate) {
        (true, true) => {
            // Stereo + needs resampling: do both at once
            debug!("converting stereo to mono and resampling from {:?} to {:?}", spec.sample_rate, out_spec.sample_rate);
            let ratio = out_spec.sample_rate as f64 / spec.sample_rate as f64;
            let num_frames = samples.len() / 2; // Number of stereo frames
            let new_len = (num_frames as f64 * ratio) as usize;
            let mut resampled = Vec::with_capacity(new_len);

            for i in 0..new_len {
                let src_pos = i as f64 / ratio;
                let src_frame = src_pos as usize;
                let src_index = src_frame * 2;

                if src_index + 3 < samples.len() {
                    // Linear interpolation between two stereo frames
                    let frac = src_pos - src_frame as f64;

                    // First stereo frame (convert to mono)
                    let mono1 = (samples[src_index] as f64 + samples[src_index + 1] as f64) / 2.0;
                    // Second stereo frame (convert to mono)
                    let mono2 = (samples[src_index + 2] as f64 + samples[src_index + 3] as f64) / 2.0;

                    // Interpolate between the two mono samples
                    let interpolated = mono1 + (mono2 - mono1) * frac;
                    resampled.push(interpolated as i16);
                } else if src_index + 1 < samples.len() {
                    // Just one stereo frame available
                    let mono = ((samples[src_index] as i32 + samples[src_index + 1] as i32) / 2) as i16;
                    resampled.push(mono);
                }
            }
            resampled
        },
        (true, false) => {
            // Stereo only: convert to mono
            debug!("converting stereo to mono");
            samples.chunks_exact(2)
                .map(|chunk| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
                .collect()
        },
        (false, true) => {
            // Mono but needs resampling
            debug!("resampling from {:?} to {:?}", spec.sample_rate, out_spec.sample_rate);
            let ratio = out_spec.sample_rate as f64 / spec.sample_rate as f64;
            let new_len = (samples.len() as f64 * ratio) as usize;
            let mut resampled = Vec::with_capacity(new_len);

            for i in 0..new_len {
                let src_pos = i as f64 / ratio;
                let src_index = src_pos as usize;

                if src_index + 1 < samples.len() {
                    // Linear interpolation
                    let frac = src_pos - src_index as f64;
                    let sample1 = samples[src_index] as f64;
                    let sample2 = samples[src_index + 1] as f64;
                    let interpolated = sample1 + (sample2 - sample1) * frac;
                    resampled.push(interpolated as i16);
                } else if src_index < samples.len() {
                    resampled.push(samples[src_index]);
                }
            }
            resampled
        },
        (false, false) => {
            // Already mono and correct sample rate
            samples
        }
    }
}

// This function was AI-generated; this should be reviewed.
fn resample_audio(wav_path: &PathBuf, wav: &mut WavReader<File>, spec: WavSpec) -> Result<Vec<i16>, Box<dyn Error>> {
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8) => {
            // 8-bit int: read as i8 and scale up to i16
            // 8-bit audio is typically unsigned (0-255, centered at 128)
            Ok(wav.samples::<i16>().collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read 8-bit samples from {:?}: {}", wav_path, e))?
                .into_iter()
                .map(|s| s << 8)  // Scale from 8-bit to 16-bit range
                .collect())
        },
        (SampleFormat::Int, 16) => {
            // 16-bit int: read directly
            wav.samples::<i16>().collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read 16-bit samples from {:?}: {}", wav_path, e).into())
        },
        (SampleFormat::Int, bits) if bits > 16 && bits <= 32 => {
            // 24-bit or 32-bit int: read as i32 and scale down to i16
            Ok(wav.samples::<i32>().collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read {}-bit samples from {:?}: {}", bits, wav_path, e))?
                .into_iter()
                .map(|s| (s >> (bits - 16)) as i16)
                .collect())
        },
        (SampleFormat::Float, _) => {
            // Float samples: read as f32 and convert to i16 range
            Ok(wav.samples::<f32>().collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read float samples from {:?}: {}", wav_path, e))?
                .into_iter()
                .map(|s| (s * i16::MAX as f32) as i16)
                .collect())
        },
        _ => {
            Err(format!("Unsupported sample format {:?} with {} bits for {:?}",
                        spec.sample_format, spec.bits_per_sample, wav_path).into())
        }
    }
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
                Ok(specs) => {
                    for spec in specs {
                        // Calculate natural width for 2-second window: (88200 - 2048) / 1024 + 1 ≈ 85
                        let window_samples = 2 * 44100;  // 2 seconds at 44100 Hz
                        let fft_size = 2048;
                        let hop_size = fft_size / 2;
                        let natural_width = ((window_samples - fft_size) / hop_size + 1).max(1);

                        // Grayscale spectrogram: [height, width] = [bands, natural_width]
                        let tensor_2d = Tensor::<B, 2>::from_data(
                            TensorData::from_bytes_vec(spec, [self.bands, natural_width], DType::U8).convert::<f32>(),
                            device
                        );
                        // Transpose to [width, height] = [natural_width, bands] for LSTM (time, freq)
                        let transposed = tensor_2d.swap_dims(0, 1);
                        // Reshape to [1, seq_len, features] = [1, natural_width, bands] for LSTM input
                        let tensor_3d = transposed.reshape([1, natural_width, self.bands]);
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

        let wavs = Tensor::cat(wavs, 0);
        let targets = Tensor::cat(targets, 0);
        debug!("wavs shape: {:?}, targets shape: {:?}", wavs.shape(), targets.shape());
        UrbanSoundBatch { sonograms: wavs, targets }
    }
}