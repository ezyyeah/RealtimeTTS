import os
import wave
import io
import pyaudio
import shutil
from typing import Optional, TYPE_CHECKING
from .base_engine import BaseEngine
from queue import Queue

# Use TYPE_CHECKING for the import so type checkers see it,
# but it doesn't cause a runtime error if piper-python isn't installed.
if TYPE_CHECKING:
    from piper.voice import PiperVoice as PiperPythonVoice

# Runtime check remains necessary
try:
    from piper.voice import PiperVoice as PiperPythonVoiceRuntime
except ImportError:
    PiperPythonVoiceRuntime = None
    # Don't print here, let the engine initialization handle the error if used.

class PiperVoice:
    """
    Represents a Piper voice configuration (model and config paths).
    Loading is handled by the PiperEngine.

    Args:
        model_file (str): Path to the Piper ONNX model (.onnx).
        config_file (Optional[str]): Path to the Piper JSON configuration file (.json).
                                     If not provided, it will be derived by appending ".json" to model_file
                                     if that file exists.
    """
    def __init__(self,
                 model_file: str,
                 config_file: Optional[str] = None):
        self.model_file = model_file

        # Determine config file path
        if config_file is None:
            possible_json = f"{model_file}.json"
            self.config_file = possible_json if os.path.isfile(possible_json) else None
        else:
            self.config_file = config_file

        # Basic validation
        if not os.path.isfile(self.model_file):
             # Warn or raise? Let's raise here for clarity. Engine will catch later if needed.
             raise FileNotFoundError(f"Model file not found: {self.model_file}")
        if self.config_file and not os.path.isfile(self.config_file):
             raise FileNotFoundError(f"Specified config file not found: {self.config_file}")


    def __repr__(self):
        return (
            f"PiperVoice(model_file={self.model_file}, "
            f"config_file={self.config_file})"
        )


class PiperEngine(BaseEngine):
    """
    A real-time text-to-speech engine using the piper-python library.
    Loads the model specified by the PiperVoice instance during initialization.
    """

    def __init__(self,
                 voice: PiperVoice,
                 use_cuda: bool = False # Added option to control GPU usage
                 ):
        """
        Initializes the Piper text-to-speech engine and loads the specified model.

        Args:
            voice (PiperVoice): A PiperVoice instance containing model/config paths.
            use_cuda (bool): Whether to attempt loading the model onto CUDA.

        Raises:
            ImportError: If piper-python library is not installed.
            FileNotFoundError: If the model or config file specified in PiperVoice is not found.
            Exception: For other errors during model loading.
        """
        if PiperPythonVoiceRuntime is None:
             raise ImportError("piper-python library is required but not found. Please install it: pip install piper-tts")

        if not isinstance(voice, PiperVoice):
            raise TypeError("Please provide a PiperVoice instance.")

        self.voice_config = voice # Store the configuration
        self.loaded_piper_voice: Optional['PiperPythonVoice'] = None # Initialize loaded voice state

        print(f"Initializing PiperEngine: Loading model '{self.voice_config.model_file}'...")
        try:
            # Load the model using paths from the voice config
            self.loaded_piper_voice = PiperPythonVoiceRuntime.load(
                self.voice_config.model_file,
                config_path=self.voice_config.config_file,
                use_cuda=use_cuda
            )
            print("Piper model loaded successfully.")
        except FileNotFoundError as e:
             print(f"Error loading Piper model: File not found - {e}")
             raise e # Re-raise the specific error
        except Exception as e:
            print(f"Error loading Piper model: {e}")
            raise e # Re-raise other loading errors

        self.queue = Queue()
        self.post_init()

    def post_init(self):
        self.engine_name = "piper_python"

    def get_stream_info(self):
        """
        Returns PyAudio stream configuration based on the loaded Piper model's sample rate.

        Returns:
            tuple: (format, channels, rate)

        Raises:
            RuntimeError: If the model wasn't loaded successfully during init.
        """
        if not self.loaded_piper_voice:
             # This shouldn't happen if __init__ succeeded, but good practice to check
             raise RuntimeError("Piper model is not loaded. Engine initialization might have failed.")

        rate = self.loaded_piper_voice.config.sample_rate
        channels = 1 # Piper models are typically mono
        audio_format = pyaudio.paInt16 # Piper outputs 16-bit PCM

        return audio_format, channels, rate

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text into audio data using the loaded Piper model.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.loaded_piper_voice:
            print("Error: Piper model is not loaded.")
            return False

        try:
            # Synthesize audio stream directly using the loaded voice
            audio_stream = self.loaded_piper_voice.synthesize_stream_raw(text)

            # Collect all chunks from the stream
            audio_bytes = b"".join(audio_stream)

            if not audio_bytes:
                 print("Warning: Piper synthesis returned no audio data.")
                 # Should this be False? If piper returns nothing, it's arguably not successful.
                 return False

            # Put the raw audio bytes onto the queue
            self.queue.put(audio_bytes)

            return True

        except Exception as e:
            print(f"Error during Piper synthesis: {e}")
            return False

    def get_voices(self):
        """
        Piper doesn't provide a way to list available voices dynamically through the library.
        This method returns an empty list. Users should manage their model files.

        Returns:
            list: Empty list.
        """
        return []
