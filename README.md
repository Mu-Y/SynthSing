# SynthSing
This is the course project for USC course EE599: Deep Learning Lab for Speech Processing. Full report is here

In this project, we implemented SynthSing, a parametric singing voice synthesizer. We managed to partially replicate this paper [A Neural Parametric Singing Synthesizer Modeling Timbre and Expression from Natural Songs](https://www.mdpi.com/2076-3417/7/12/1313) by Merlijn Blaauw and Jordi Bonada from Universitat Pompeu Fabra. Although this is not a complete replication of their paper, we believe the core modeling part, i.e. WaveNet with local conditioniong has been successfully implemented and will make it easier to replicate other modeling parts of their paper.

We borrowed a lot from [this](https://github.com/ibab/tensorflow-wavenet) implementation for the Vanilla sample-to-sample WaveNet. However, since no local conditionaing(e.g. F0 and phoneme identity for melodic and phonetic modeling) was implemented, which is necessary for meaningful synthesis, **we implemented local conditioning** by ourselves. There are also many adaptions to the Vanilla WaveNet architecture, referring to NPSS paper.

**Long story short**, our model makes frame-to-frame predictions on 60-dimensional MFSCs coefficients and 4-dimensional Aperiodicity coefficients, using F0(coarse coded), phoneme identity(one-hot coded) and normalized phoneme position(coarse coded) as local control inputs and singer identity as global control inputs. Then we feed generated MFSCs and APs, as well as true F0 into WORLD vocoder to synthesize audio.

## Prerequisite
Apart from the requirements presented [here](https://github.com/ibab/tensorflow-wavenet), you also need `pysptk` and `pyworld` for audio resynthsis and data pre-processing.

## Dataset and Pre-trained Models
We used two datasets: 1) NIT Japanese Nursery and 2) self-curated Coldplay songs, using [GENTLE](https://github.com/lowerquality/gentle). 

You can download our pre-processed(i.e. time-aligned and transformed) dataset for NIT Japanese Nursery, including MFSC, coarse coded F0, AP, etc [here](https://drive.google.com/open?id=1LQgP49jjZTb4FVEf5PevsoiBDhCIyaWp). It also contains our pre-trained models which can be directly used for generating NIT Japanese Nursery singing. After unzipping, put folder `data` under `SynthSing/`, `worked_model_mfsc` under `SynthSing/WaveNet-Harm/`, `worked_model_ap` under `SynthSing/WaveNet-Aper/` respectively.

## Experiments
Run our pre-trained models and generate MFSCs and APs:
```
# Generate the first 3200 frames of MFSCs using the pre-trained Harmonic model 
# for the target clip nitech_jp_song070_f001_004, using reference F0 and phonemes. 
SynthSing$ cd WaveNet-Harm
WaveNet-Harm$ python generate.py --frames 3200 --clip_id 004 --wav_out_path generated_004.npy --gc_channels=32 --gc_cardinality=2 --gc_id=1 ./worked_model_mfsc/model.ckpt-1894500

# Generate the first 3200 frames of APs using the pre-trained Aperiodicity model 
# for the target clip nitech_jp_song070_f001_004, using reference F0 and phonemes, as well as generated MFSCs. 
SynthSing$ cd WaveNet-Aper
WaveNet-Aper$ python generate.py --frames 3200 --clip_id 004 --wav_out_path generated_004.npy --gc_channels=32 --gc_cardinality=2 --gc_id=1 ./worked_model_ap/model.ckpt-478500

# resynthesize audio using the generated MFSCs and APs, as well as reference F0 via WORLD vocoder
SynthSing$ python resynth.py --mfsc_file WaveNet-Harm/generated_004.npy --f0_file data/f0ref/nitech_jp_song070_f001_004.npy --ap_file WaveNet-Aper/generated_004.npy --wave_out_path resynth_004.wav

# For our hand-crafted scrambled sequence
WaveNet-Harm$ python generate.py --frames 1770 --scramble --wav_out_path generated_scramble_mfsc.npy --gc_channels=32 --gc_cardinality=2 --gc_id=1 ./worked_model_mfsc/model.ckpt-1894500
WaveNet-Aper$ python generate.py --frames 1770 --scramble --wav_out_path generated_scramble_ap.npy --gc_channels=32 --gc_cardinality=2 --gc_id=1 ./worked_model_ap/model.ckpt-478500
SynthSing$ python resynth.py --mfsc_file WaveNet-Harm/generated_scramble_mfsc.npy --f0_file data/F0ref_scramble.npy --ap_file WaveNet-Aper/generated_scramble_ap.npy --wav_out_path resynth_scramble.wav
```
Train new models:
```
# Same command for both Harmonic and Aperiodicity model
python train.py --gc_channels=32
```


## Results (audio samples)
- Trained on NIT data. Replicating results from the NIT dataset. We took one of the training recordings as target. Resynthesized using true F0, generated MFSC and AP.
  - [Target](https://soundcloud.com/mu-yang-974011976/hit-004_orignal?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/hit_004_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)
- Trained on NIT data. Generating previously unseen sequences by splicing together random clips from the NIT recordings and doing a similar concatenation of the corresponding F0 and phonemes for each audio clip.
  - [Target](https://soundcloud.com/mu-yang-974011976/hit_scramble_original?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/hit_scramble_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)
- Trained on self-created dataset. we resynthesized recordings in the Coldplay dataset using true F0 and AP, and MFSCs generated by the harmonic submodel.
  - [Target](https://soundcloud.com/mu-yang-974011976/coldplay-song02-01-007?in=mu-yang-974011976/sets/results-for-synthsing)
  - [Synthesized](https://soundcloud.com/mu-yang-974011976/coldplay_007_synthesized?in=mu-yang-974011976/sets/results-for-synthsing)

