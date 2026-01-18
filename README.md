# lstm-example

This is a simple toy program for training/predicting audio
sample features using Rust + Burn.

It started as a port of [Audio_Classification_using_LSTM](https://github.com/sarthak268/Audio_Classification_using_LSTM)
to Rust so I could learn more Rust and machine learning on Rust.

I'm using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset to train
and test, if you want to try it out you'll need to download that dataset yourself!

To preprocess audio: normalize all audio to 44.1k mono floating point.

```shell
cd path/to/UrbanSound8K
find audio -name '*.wav' -exec sox --channels 1 --bits 32 --encoding floating-point --rate 44100 norm_{} \;
```
