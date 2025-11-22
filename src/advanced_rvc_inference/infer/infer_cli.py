from assets.logging_config import configure_logging
from assets.model_installer import check_and_install_models

configure_logging(True, False, "WARNING")
check_and_install_models()

import argparse
from distutils.util import strtobool

from rvc.infer.infer import rvc_edgetts_infer, rvc_infer


def create_parser():
    # Base parser with common arguments
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--rvc_model", type=str, required=True, help="Name of the RVC model")
    base.add_argument("--f0_method", type=str, default="rmvpe", help="F0 extraction method")
    base.add_argument("--f0_min", type=int, default=50, help="Minimum F0 frequency")
    base.add_argument("--f0_max", type=int, default=1100, help="Maximum F0 frequency")
    base.add_argument("--hop_length", type=int, default=128, help="Hop length for Crepe processing")
    base.add_argument("--rvc_pitch", type=float, default=0, help="Pitch of the RVC model")
    base.add_argument("--protect", type=float, default=0.5, help="Consonant protection")
    base.add_argument("--index_rate", type=float, default=0, help="Index rate")
    base.add_argument("--volume_envelope", type=float, default=1, help="Volume envelope")
    base.add_argument("--autopitch", type=lambda x: bool(strtobool(x)), default=False, help="Automatic pitch detection")
    base.add_argument("--autopitch_threshold", type=float, default=155.0, help="155.0 - Male model | 255.0 - Female model")
    base.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=False, help="Pitch correction")
    base.add_argument("--autotune_strength", type=float, default=1.0, help="Autotune strength")
    base.add_argument("--output_format", type=str, default="mp3", help="Output file format")

    # Main parser with subcommands
    parser = argparse.ArgumentParser(description="Tool for voice replacement using RVC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand for RVC
    rvc = subparsers.add_parser("rvc", parents=[base], help="Convert an audio file")
    rvc.add_argument("--input_path", type=str, required=True, help="Path to the audio file")

    # Subcommand for TTS
    edge_tts = subparsers.add_parser("tts", parents=[base], help="Synthesize speech from text")
    edge_tts.add_argument("--tts_voice", type=str, required=True, help="Voice for speech synthesis")
    edge_tts.add_argument("--tts_text", type=str, required=True, help="Text for speech synthesis")
    edge_tts.add_argument("--tts_rate", type=int, default=0, help="Speech synthesis rate")
    edge_tts.add_argument("--tts_volume", type=int, default=0, help="Speech synthesis volume")
    edge_tts.add_argument("--tts_pitch", type=int, default=0, help="Speech synthesis pitch")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    common_params = {
        "rvc_model": args.rvc_model,
        "f0_method": args.f0_method,
        "f0_min": args.f0_min,
        "f0_max": args.f0_max,
        "hop_length": args.hop_length,
        "rvc_pitch": args.rvc_pitch,
        "protect": args.protect,
        "index_rate": args.index_rate,
        "volume_envelope": args.volume_envelope,
        "autopitch": args.autopitch,
        "autopitch_threshold": args.autopitch_threshold,
        "autotune": args.autotune,
        "autotune_strength": args.autotune_strength,
        "output_format": args.output_format,
    }

    if args.command == "rvc":
        rvc_infer(**common_params, input_path=args.input_path)
    elif args.command == "tts":
        rvc_edgetts_infer(
            **common_params,
            tts_voice=args.tts_voice,
            tts_text=args.tts_text,
            tts_rate=args.tts_rate,
            tts_volume=args.tts_volume,
            tts_pitch=args.tts_pitch,
        )

    print("\033[1;92m\nVoice successfully replaced!\n\033[0m")


if __name__ == "__main__":
    main()
