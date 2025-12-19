use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::prelude::{Backend, ElementConversion, Int, TensorData};
use burn::Tensor;
use burn::tensor::DType;
use gif::{ColorOutput, DecodeOptions, Decoder, Encoder, Frame};
use hound::{SampleFormat, WavReader, WavSpec};
use serde::Deserialize;
use sonogram::{ColourGradient, ColourTheme, FrequencyScale, SpecOptionsBuilder};
use log::{debug, info, warn};

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
}

impl UrbanSoundDataset {
    pub fn new(path: &PathBuf) -> Result<Self, std::io::Error> {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');
        let csv_path = path.join("metadata").join("UrbanSound8K.csv");
        let dataset = InMemDataset::from_csv(csv_path, rdr)?;
        let dataset = UrbanSoundDataset { path: path.clone(), dataset };
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
    debug!("opened handle {:?}", handle);
    let mut wav = WavReader::new(handle)
        .map_err(|e| format!("Failed to read WAV {:?}: {}", wav_path, e))?;
    let spec = wav.spec();
    info!("read wav {:?}: channels={}, sample_rate={}, bits={}, format={:?}",
           wav_path.file_name().unwrap_or_default(), spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format);
    let out_spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    // Read all samples from the WAV file and convert to i16
    // Handle different bit depths
    let samples: Vec<i16> = resample_audio(wav_path, &mut wav, spec)?;
    debug!("samples {:?}", samples.len());

    // Convert to mono and resample in one pass if needed
    let resampled_samples = convent_to_mono(spec, out_spec, samples);

    // Calculate window and step sizes in samples (using output sample rate)
    let window_duration_secs = 2;
    let step_duration_secs = 1;
    let window_size = window_duration_secs * out_spec.sample_rate as usize;
    let step_size = step_duration_secs * out_spec.sample_rate as usize;

    debug!("resampling/conversion done; window size {}, step size {}", window_size, step_size);

    // Pad short audio files with silence to reach minimum window size
    let resampled_samples = if resampled_samples.len() < window_size {
        debug!("padding audio from {} to {} samples", resampled_samples.len(), window_size);
        let mut padded = resampled_samples;
        padded.resize(window_size, 0);
        padded
    } else {
        resampled_samples
    };

    let mut result = Vec::new();
    let mut gradient = ColourGradient::create(ColourTheme::Default);

    // Iterate over 2-second windows with 1-second offset
    let mut offset = 0;
    while offset + window_size <= resampled_samples.len() {
        let precomputed_path = wav_path.with_added_extension(format!("precomputed_{}_{}.gif", window_size, offset));
        debug!("pulling out window [{}..{}] of {}", offset, offset + window_size, resampled_samples.len());
        let window_samples = &resampled_samples[offset..offset + window_size];

        debug!("creating sonogram at offset {}, window_samples len: {}, bands: {}", offset, window_samples.len(), bands);

        // Create spectrogram from the window
        debug!("loading data from memory...");

        let gif_path = if precomputed_path.exists() {
            Some(precomputed_path.clone())
        } else {
            None
        };

        let grayscale_bytes = match gif_path {
            Some(path) => {
                info!("loading sonogram data from file {}", path.display());
                read_gif_file(&path)?
            }
            None => {
                info!("generating sonogram data for file {}", wav_path.display());
                let (spec, width, height) = sample_to_spectrogram(bands, wav_path, out_spec, &mut gradient, offset, window_samples)?;
                // Save the computed spectrogram
                let file = File::create(precomputed_path)?;
                let mut encoder = Encoder::new(file, width as u16, height as u16, &[0xff, 0xff, 0xff, 0, 0, 0])?;
                let frame = Frame::from_indexed_pixels(width as u16, height as u16, spec.clone(), None);
                encoder.write_frame(&frame)?;
                spec
            }
        };

        debug!("converted to grayscale, len: {}", grayscale_bytes.len());
        result.push(grayscale_bytes);
        offset += step_size;
    }

    Ok(result)
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