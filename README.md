# Chord Sequence Generator

This repository contains tools for generating chord sequences that match the melody of audio inputs. The project was developed during the Generative AI Workshop at UPF Barcelona and Sound of AI.

## Approaches

### Stochastic Methods
- Markov Chain Modeling
- Chord Classifier Sampling

Using the [Choco Dataset](https://github.com/smashub/choco.git), we:
- Developed a parser to convert Harte annotations to MIDI
- Created n-order Markov transition matrices from unique chords
- Extracted melodies using Spotify Basic Pitch and Essentia Melodia
- Generated harmonies matching the detected melodies

### Generative Model
- Variational Auto Encoder (VAE) that compresses Choco JAM data into latent representations for chord sampling

## Features
- Max/MSP patch with OSC communication
- Automatic generation when new audio files are detected

*Note: This is an exploratory project evaluating different approaches to chord sequence generation.*

## Setup Requirements

### Dependencies
- Python 3.11 (required)
- basic-pitch
- librosa
- numpy

### Installation Options

#### Using Conda
```bash
conda create -n genmus_bcn2024 python=3.11
conda activate genmus_bcn2024
pip install basic-pitch librosa numpy
```

_Note: Basic Pitch requires Tensorflow and Python 3.11 for proper dependency installation._

#### Using Pip

`pip install -r requirements.txt`

#### Workflow

1. Clone the Choco Dataset into the `/data` folder
2. Run markov_sequence_generator.py to build transition matrices from the dataset
3. Place your target audio loop in the `/data/loop` folder
4. Execute the main script to generate chord sequences for your input
5. Find the generated MIDI sequences in the `/data/midi` folder