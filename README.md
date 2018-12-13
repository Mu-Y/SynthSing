# SynthSing
This is the course project for USC course EE599: Deep Learning Lab for Speech Processing. Full report is here

In this project, we implemented SynthSing, a parametric singing voice synthesizer. We managed to replicate this [Neural Parametric Singing Synthesizer(NPSS)](https://www.dtic.upf.edu/~mblaauw/NPSS/) by Merlijn Blaauw and Jordi Bonada from Universitat Pompeu Fabra. Their original paper is [here](https://www.mdpi.com/2076-3417/7/12/1313). Since they did not make their code public, we implemented their paper by ourselves. 

We borrowed a lot from [this](https://github.com/ibab/tensorflow-wavenet) implementation for the Vanilla sample-to-sample WaveNet. However, since no local conditionaing(e.g. F0 and phoneme identity for melodic and phonetic modeling) was implemented, which is necessary for meaningful synthesis, we also implemented it by ourselves. There are also many adaptions to the Vanilla WaveNet architecture, referring to NPSS paper.

**Long story short**, our model makes frame-to-frame predictions on 60-dimensional MFSCs and 4-dimensional APs, using F0(coarse coded), phoneme identity(one-hot coded) and normalized phoneme position(coarse coded) as local control inputs and singer identity as global control inputs. Then we feed generated MFSCs and APs, as well as true F0 into WORLD vocoder to synthesize audio.

## Dataset
We trained our model on two datasets: 
- [NIT Japanese Nursery](http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2). This is what was used in NPSS paper. We used this because this is the only publicly available phoneme-level aligned dataset that we found.
- Our self-created dataset - a collection of 10 Coldplay songs with isolated lead vocal tracks downloaded from YouTube. The phoneme-level alignment were generated from the force aligner [GENTLE](https://github.com/lowerquality/gentle). Due to the limitation of GENTLE's recognition capability and the available Acapella audio, this dataset is relatively small.

## Results (audio samples)
- Trained on NIT data. Replicating results from the NIT dataset. We took one of the training recordings as target. Resynthesized using true F0, generated MFSC and AP.
  - [Target](https://soundcloud.com/mu-yang-974011976/hit-004_orignal?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/hit_004_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)
- Trained on NIT data. Generating previously unseen sequences. By splicing together random clips from the NIT recordings and doing a similar concatenation of the corresponding F0 and phonemes for each audio clip, we created control input sequences that the model had not seen before. Using the grafted control inputs, we generated MFSCs and AP coefficients from the coarse-coded F0s and phonetic timings and then re-synthesized the audio via WORLD.
  - [Target](https://soundcloud.com/mu-yang-974011976/hit_scramble_original?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/hit_scramble_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)
- Trained on self-created dataset. we resynthesized recordings in the Coldplay dataset using true F0 and AP, and MFSCs generated by the harmonic submodel.
  - [Target](https://soundcloud.com/mu-yang-974011976/coldplay-song02-01-007?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/coldplay_007_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)

## Adaption to the original NPSS paper
The system we originally intended to implement contained all three models of NPSS - the timbre model, pitch model and phonetic timing model. However, due to the challenges of implementing a WaveNet model with local conditioning and constructing a dataset from scratch, we were only able to build and train the timbre model and its two submodels - the harmonic model and aperiodic model. As we used F0 values from a recording, we did not need to implement the voiced/unvoiced submodel.

However, we incorporated global conditioning in our WaveNet architecture, which was not mentioned in the vanilla NPSS paper. This gives our model the potential for transfer learning task - given data for several new singers, our model is able to synthesize singing for those singers, or even model cross-singer singing voice, e.g. having “Taylor Swift” sing like “Ed Sheeran” given a few lyrics.