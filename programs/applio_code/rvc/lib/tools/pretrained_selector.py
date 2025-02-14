def pretrained_selector(pitch_guidance):
    if pitch_guidance == True:
        return {
            "v1": {
                32000: (
                    "rvc/models/pretraineds/pretrained_v1/f0G32k.pth",
                    "rvc/models/pretraineds/pretrained_v1/f0D32k.pth",
                ),
                40000: (
                    "rvc/models/pretraineds/pretrained_v1/f0G40k.pth",
                    "rvc/models/pretraineds/pretrained_v1/f0D40k.pth",
                ),
                48000: (
                    "rvc/models/pretraineds/pretrained_v1/f0G48k.pth",
                    "rvc/models/pretraineds/pretrained_v1/f0D48k.pth",
                ),
            },
            "v2": {
                32000: (
                    "rvc/models/pretraineds/pretrained_v2/f0G32k.pth",
                    "rvc/models/pretraineds/pretrained_v2/f0D32k.pth",
                ),
                40000: (
                    "rvc/models/pretraineds/pretrained_v2/f0G40k.pth",
                    "rvc/models/pretraineds/pretrained_v2/f0D40k.pth",
                ),
                48000: (
                    "rvc/models/pretraineds/pretrained_v2/f0G48k.pth",
                    "rvc/models/pretraineds/pretrained_v2/f0D48k.pth",
                ),
            },
        }
    elif pitch_guidance == False:
        return {
            "v1": {
                32000: (
                    "rvc/models/pretraineds/pretrained_v1/G32k.pth",
                    "rvc/models/pretraineds/pretrained_v1/D32k.pth",
                ),
                40000: (
                    "rvc/models/pretraineds/pretrained_v1/G40k.pth",
                    "rvc/models/pretraineds/pretrained_v1/D40k.pth",
                ),
                48000: (
                    "rvc/models/pretraineds/pretrained_v1/G48k.pth",
                    "rvc/models/pretraineds/pretrained_v1/D48k.pth",
                ),
            },
            "v2": {
                32000: (
                    "rvc/models/pretraineds/pretrained_v2/G32k.pth",
                    "rvc/models/pretraineds/pretrained_v2/D32k.pth",
                ),
                40000: (
                    "rvc/models/pretraineds/pretrained_v2/G40k.pth",
                    "rvc/models/pretraineds/pretrained_v2/D40k.pth",
                ),
                48000: (
                    "rvc/models/pretraineds/pretrained_v2/G48k.pth",
                    "rvc/models/pretraineds/pretrained_v2/D48k.pth",
                ),
            },
        }
