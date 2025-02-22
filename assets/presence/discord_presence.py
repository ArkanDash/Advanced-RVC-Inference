from pypresence import Presence
from pypresence.exceptions import DiscordNotFound, InvalidPipe
import datetime as dt
import threading
import functools


class RichPresenceManager:
    def __init__(self):
        self.client_id = "1340358329771364430"
        self.rpc = None
        self.running = False
        self.current_state = "Idling"
        self.lock = threading.Lock()
        self.discord_available = True

    def start_presence(self):
        try:
            if not self.running:
                self.rpc = Presence(self.client_id)
                try:
                    self.rpc.connect()
                    self.running = True
                    self.discord_available = True
                    self.update_presence()
                    print("Discord Rich Presence connected successfully")
                except (DiscordNotFound, InvalidPipe):
                    print("Discord is not running. Rich Presence will be disabled.")
                    self.discord_available = False
                    self.running = False
                    self.rpc = None
                except Exception as error:
                    print(f"An error occurred connecting to Discord: {error}")
                    self.discord_available = False
                    self.running = False
                    self.rpc = None
        except Exception as e:
            print(f"Unexpected error in start_presence: {e}")
            self.discord_available = False
            self.running = False
            self.rpc = None

    def update_presence(self):
        if self.rpc and self.running and self.discord_available:
            try:
                config = self.get_presence_config(self.current_state)
                self.rpc.update(
                    state=self.current_state,
                    details="Advanced RVC Inference UI",
                    buttons=[
                        {
                            "label": "Download",
                            "url": "https://github.com/ArkanDash/Advanced-RVC-Inference.git",
                        }
                    ],
                    large_image="logo",
                    large_text="Advanced RVC Inference for quicker and effortless model downloads",
                    small_image=config["small_image"],
                    small_text=config["small_text"],
                    start=dt.datetime.now().timestamp(),
                )
            except Exception as e:
                print(f"Error updating Discord presence: {e}")
                self.discord_available = False
                self.cleanup()

    def set_state(self, state):
        if self.discord_available:
            with self.lock:
                self.current_state = state
                if self.running:
                    self.update_presence()

    def cleanup(self):
        self.running = False
        if self.rpc and self.discord_available:
            try:
                self.rpc.close()
            except:
                pass
        self.rpc = None
        self.discord_available = False

    def stop_presence(self):
        self.cleanup()


RPCManager = RichPresenceManager()


def track_presence(state_message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if RPCManager.running and RPCManager.discord_available:
                RPCManager.set_state(state_message)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if RPCManager.running and RPCManager.discord_available:
                    RPCManager.set_state("Idling")

        return wrapper

    return decorator
