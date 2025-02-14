import sys
import asyncio
import edge_tts


async def main():
    # Parse command line arguments
    text = str(sys.argv[1])
    voice = str(sys.argv[2])
    rate = int(sys.argv[3])
    output_file = str(sys.argv[4])

    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"

    await edge_tts.Communicate(text, voice, rate=rates).save(output_file)
    print(f"TTS with {voice} completed. Output TTS file: '{output_file}'")


if __name__ == "__main__":
    asyncio.run(main())
