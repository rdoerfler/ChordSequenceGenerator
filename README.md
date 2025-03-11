### Generative AI Music Workshop

Authors: Robin Doerfler, Michael Taenzer

This repository contains scripts and pipelines for chord sequence generation given the melody of an input audio file.
This exploration was conducted within the context of the Generative AI Workshop at UPF Barcelona and Sound of AI.

The task was approached in multiple ways:
- by storchastic processes (Markov Chain Modeling) & Sampling from a Chord Classifier.
- by probabilistic generative models (Variational Auto Encoder).

For the stochastic methods the [Choco Dataset](https://github.com/smashub/choco.git) was used as a corpus.
We implemented a parser from the harte annotation (choco) to midi objects as a more convenient way of interacting in python.
First we reduce the corpus to only contain unique chords.
We then compute n-order markov transition matrices based on the whole corpus and their chord sequences.
We use the Spotify Basic Pitch Network as well as the Essentia Melodia Pitch detection module for derive f0 sequences. 
Based on the estimated predominant melody, we start the markov sequence generation, to produce a harmony fitting the context of the detected melody.

The second approach uses a Variational Auto Encoder Network. 
The choco jams data is compressed into a latent representation from which new chords can be sampled.

An utility max/msp patch and osc receiver & streamer allows to trigger new loop generations whenever a new input audio file was recorded to a specified folder.

Note: This work is merely an exploration and evaluation of various approaches to a complex generative task.

These scripts should be run in a conda environment with the following requirements:

1. python 3.11 (important!)  
2. basic-pitch  
3. librosa  
4. numpy  

Installation (Conda):  
<code>conda create -n genmus_bcn2024 python=3.11
pip install basic-pitch
pip install librosa
pip install numpy
conda activate genmus_bcn2024
</code>

(Basic Pitch needs Tensorflow and Python 3.11 to install the correct dependencies.)

Installation (pip):

`pip install -r requirements.txt`

Usage:
1. Drop [Choco Dataset](https://github.com/smashub/choco.git) into **/data** folder.
2. Run **markov_sequence_generator.py** to prepare markov transition matrices once on the full dataset.
3. Drop input target loop into **data/loop** folder.
4. Run **main script** to generate new chord sequences based on input loop.
5. Find generated MIDI Sequence in **data/midi** folder.