from pypresence import Presence
import datetime as dt


class RichPresenceManager:
    def __init__(self):
        self.client_id = "1340358329771364430"
        self.rpc = None
        self.running = False

    def start_presence(self):
        if not self.running:
            self.running = True
            self.rpc = Presence(self.client_id)
            try:
                self.rpc.connect()
                self.update_presence()
            except KeyboardInterrupt as error:
                print(error)
                self.rpc = None
                self.running = False
            except Exception as error:
                print(f"An error occurred connecting to Discord: {error}")
                self.rpc = None
                self.running = False

    def update_presence(self):
        if self.rpc:
            self.rpc.update(
                state="Advanced-RVC",
                details="Advaced voice cloning with UVR5 feature",
                buttons=[
                    {"label": "Home", "url": "https://github.com/ArkanDash/Advanced-RVC-Inference"},
                    {"label": "Download", "url": "https://github.com/ArkanDash/Advanced-RVC-Inference/archive/refs/heads/master.zip"},
                ],
                large_image="logo",
                large_text="Experimenting with Advanced-RVC",
                start=dt.datetime.now().timestamp(),
            )

    def stop_presence(self):
        self.running = False
        if self.rpc:
            self.rpc.close()
            self.rpc = None


RPCManager = RichPresenceManager()
