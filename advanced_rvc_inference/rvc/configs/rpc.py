import os
import sys
import json
import time
import struct
import codecs

sys.path.append(os.getcwd())

from main.app.variables import translations

CLIENT_ID = "1392816674159202396"

def create_payload(opcode, payload):
    data = json.dumps(payload).encode("utf-8")

    return struct.pack(
        "<I", 
        opcode
    ) + struct.pack(
        "<I", 
        len(data)
    ) + data

def connect_discord_ipc():
    try:
        return open(
            r"\\?\pipe\discord-ipc-0", 
            "r+b", 
            buffering=0
        )
    except Exception:
        return None

def send_discord_rpc(pipe):
    pipe.write(
        create_payload(
            0, {
                "v": 1, 
                "client_id": CLIENT_ID
            }
        )
    )

    pipe.read(8)
    pipe.read(
        struct.unpack(
            "<I", 
            pipe.read(4)
        )[0]
    )

    pipe.write(
        create_payload(
            1, {
                "cmd": "SET_ACTIVITY",
                "args": {
                    "pid": os.getpid(),
                    "activity": {
                        "buttons": [{
                            "label": "Github", 
                            "url": codecs.decode("uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP", "rot13")
                        }],
                        "details": translations["details"],
                        "timestamps": {
                            "start": int(
                                time.time()
                            )
                        },
                        "state": translations["use"]
                    }
                },
                "nonce": str(
                    time.time()
                )
            }
        )
    )