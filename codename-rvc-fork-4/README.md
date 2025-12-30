# <p align="center">` Codename-RVC-Fork 🍇 4 ` </p>
## <p align="center">Based on Applio</p>

<p align="center"> ㅤㅤ👇 You can join my discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/nQFpNBvvd3" target="_blank"> Codename's Sanctuary</a>
</p>

<p align="center"> ㅤㅤ👆 To stay up-to-date with advancements, hang out or get support 👆ㅤㅤ </p>


## A lil bit more about the project:

### This fork is pretty much my personal take on Applio. ✨
``You could say.. A more advanced features-rich Applio ~ With my lil twist.``
<br/>
``But If you have any ideas, want to pr or collaborate, feel free to do so!``
<br/>
ㅤ
<br/>
# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
`1. Datasets must be processed properly:`
- Peak or RMS compression if necessary! ( This step isn't covered by the fork's preprocessing btw.)
- Silence-truncation ( Absolutely necessary. )
- 'simple' method chosen for preprocessing ( Even 3 sec segments. )
- Enable Loudness Normalization in the ui.
- Enable automatic LUFS range finder for Loudness Normalization. <br/>
``Expect issues with PESQ and data alignment If the following requirements are not met.``


`2. Experimental things are experimental for a reason:`
- If you don't understand what it does, what it brings or how it works? preferably don't use it.
- Certain features / currently chosen params can be potentially unstable or broken and are a subject to change.
- Not all experimental features gonna reach "stable" status ( There's only as much I can test/ablation study on my own. )
- Some experimental things might disappear at some point if deemed too unstable / not worth it.***

`3. Clarification on pretrained models, architectures & vocoders:`
- **Each Architecture/Vocoder requires own dedicated pretrains.**
##### 1. HiFi-GAN ( RVC architecture ):
- The original architecture. ( HiFi-GAN + MPD, MSD )
- It's pretrained models are auto-downloaded during the first launch.
- Available for sample rates: 48, 40 and 32khz. <br/><br/>`Models made with this arch are cross-compatible: RVC, Applio and codename-rvc-fork-4.` 
##### 2. MRF-HiFi-GAN & RefineGAN ( Fork/Applio architecture ):
- Custom architecture. ( MRF-HiFi-GAN / RefineGAN + MPD, MSD )
- **Not 100% sure on the status of pretrains yet. Once I get more info, will update this entry.** <br/><br/>`Models made with this arch are LIMITED cross-compatible: codename-rvc-fork-4 and Applio`
##### 2. RingFormer ( Fork architecture ):
- Custom architecture. ( RingFormer + MPD, MSD, MRD )
- **There are no available pretrained models for it yet.**
- Planned supported sample rates: 48khz ( and *maybe* 24khz, but that's up to dr87 ).<br/><br/>`Models made with this arch ARE NOT cross-compatible: codename-rvc-fork-4` 
<br/>

# **Fork's features:**
 
- Hold-Out type validation mechanism during training. ( L1 MEL, mrSTFT, PESQ, SI-SDR )  ` In between epochs. `
 
- FP16-AMP, BF16-AMP, TF32, FP32 Training modes available.  ` BF16 & TF32 require Ampere or newer GPUs. `<br/>
`BF16 and TF32 can be used simultaneously for extra speed gains.`
> NOTE: BF16 is used by default ( and bf16-AdamW ). If unsupported hardware detected, switched back to FP32. Inference is only in FP32.

 
- Support for 'Spin' embedder.
 
- Ability to choose an optimizer.  ` ( Supported: AdamW, AdamW_BF16, RAdam, AdamSPD, Ranger21, DiffGrad, Prodigy ) `
 
- (EXP) Double-update strategy for Discriminator.
 
- Support for custom input-samples used during training for live-preview of model's reconstruction performance.
 
- Mel spectrogram %-based similarity metric.
 
- Support for Multi-scale, classic L1 mel and (EXP) multi-resolution stft spectral losses.
 
- Support for some of VITS2 enhancements.
`( Transformer-enhanced normalizing flow + spk conditioned text encoder. )`
 
- Support for the following vocoders: HiFi-GAN-NSF, MRF-HiFi-gan, Refine-GAN, RingFormer, Wavehax, Snake-HiFi-GAN-NSF.<br/>
` RingFormer, Wavehax and Snake-HiFi-GAN-NSF architectures utilize MPD, MSD and MRD Discs combo.`
 
- Checkpointing and various speed / memory optimizations compared to og RVC.
 
- New logging mechanism for losses: Average loss per epoch logged as the standard loss, <br/>and rolling average loss over 50 steps to evaluate general trends and the model's performance over time.
 
- From-ui quick tweaks; lr for g/d, schedulers, linear warmup, kl loss annealing and more ..

**Any new / experimental features are always highlighted in releases so, feel free to check it out there.**
  
 
 
 <br/>
 
 
✨ to-do list ✨
> - None
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join my discord server ❗
 <br/>
 <br/>

## Getting Started:

### 1. Installation of the Fork

Run the installation script:

- Double-click `run-install.bat`.

### 2. Running the Fork

Start Applio using:

- Double-click `run-fork.bat`.
 
This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring
 
To monitor training or visualize data:
- Run the " run_tensorboard_in_model_folder.bat " file from logs folder and paste in there path to your model's folder </br>( containing 'eval' folder or tfevents file/s. )</br></br>If it doesn't work for you due to blocked port, open up CMD with admin rights and use this command:</br>`` netsh advfirewall firewall add rule name="Open Port 25565" dir=in action=allow protocol=TCP localport=25565 ``</br></br>
- Alternatively if the above method fails, run the tensorboard manually in cmd:</br> ``tensorboard --logdir="path/to/your/model/folder" --bind_all``</br>
(PS. Make sure you have tensorboard installed. ( in cmd:  pip install tensorboard )
 
## Referenced projects
+ [RingFormer](https://github.com/seongho608/RingFormer)
+ [RiFornet](https://github.com/Respaired/RiFornet_Vocoder)
+ [bfloat_optimizer (AdamW BF16)](https://github.com/lessw2020/bfloat_optimizer)
+ [BigVGAN](https://github.com/NVIDIA/BigVGAN/tree/main)
+ [Pytorch-Snake](https://github.com/falkaer/pytorch-snake)

 
## Disclaimer
``The creators, maintainers, and contributors of the original Applio repository, as well as the creator of this fork (Codename;0), which is based on Applio, and the contributors of this fork, are not liable for any legal issues, damages, or consequences arising from the use of this repository or any content generated from it. By using this fork, you acknowledge and accept the following terms:``
 
- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md)
