"""
Audio processing utilities for VampRust Python bindings.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np


from ._vamprust import PluginInfo, VampHost, VampPlugin


class AudioProcessor:
    """High-level audio processing interface for Vamp plugins."""

    def __init__(self) -> None:
        self.host = VampHost()
        self._libraries: Dict[str, Any] = {}
        self._plugins: Dict[str, Tuple[VampPlugin, PluginInfo]] = {}
        self._cache_dir = Path.home() / ".cache" / "vamprust"
        self._plugin_cache_file = self._cache_dir / "plugin_cache.json"

    def _get_cache_key(self) -> str:
        """Generate a cache key based on library paths and timestamps."""
        library_paths = self.host.find_plugin_libraries()
        cache_data = []
        for lib_path in library_paths:
            try:
                stat = os.stat(lib_path)
                cache_data.append((str(lib_path), stat.st_mtime, stat.st_size))
            except OSError:
                cache_data.append((str(lib_path), 0, 0))
        return str(hash(str(sorted(cache_data))))

    def _load_plugin_cache(self) -> Optional[Dict[str, Any]]:
        """Load plugin information from filesystem cache."""
        if not self._plugin_cache_file.exists():
            return None

        try:
            with open(self._plugin_cache_file, "r") as f:
                cache = json.load(f)

            # Check if cache is still valid
            current_key = self._get_cache_key()
            if cache.get("cache_key") == current_key:
                return cache.get("plugins", {})
        except (json.JSONDecodeError, OSError):
            pass

        return None

    def _save_plugin_cache(self, plugins_data: Dict[str, Any]) -> None:
        """Save plugin information to filesystem cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        cache = {"cache_key": self._get_cache_key(), "plugins": plugins_data}

        try:
            with open(self._plugin_cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except OSError:
            pass  # Silently fail if can't write cache

    def discover_plugins(self) -> List[PluginInfo]:
        """Discover all available Vamp plugins."""
        # Try to load from cache first
        cached_data = self._load_plugin_cache()
        if cached_data:
            # Reconstruct plugins from cached data
            plugins = []
            for lib_path_str, plugin_data_list in cached_data.items():
                lib_path = Path(lib_path_str)
                if lib_path.exists():
                    library = self.host.load_library(str(lib_path))
                    if library:
                        self._libraries[lib_path_str] = library
                        # Get fresh plugin info from library (can't serialize PluginInfo objects)
                        plugins.extend(library.list_plugins())
            return plugins

        # Cache miss - do full discovery
        plugins = []
        library_paths = self.host.find_plugin_libraries()
        plugins_data = {}

        for lib_path in library_paths:
            library = self.host.load_library(lib_path)
            if library:
                self._libraries[lib_path] = library
                plugin_list = library.list_plugins()
                plugins.extend(plugin_list)

                # Store basic plugin info for caching (identifiers only)
                plugins_data[lib_path] = [
                    {"identifier": p.identifier, "name": p.name, "index": p.index}
                    for p in plugin_list
                ]

        # Save to cache
        self._save_plugin_cache(plugins_data)
        return plugins

    def fuzzy_search_plugins(
        self, query: str, threshold: int = 60, limit: int = 10
    ) -> List[PluginInfo]:
        """Fuzzy search for plugins by name or identifier."""
        if not self._libraries:
            self.discover_plugins()
        from rapidfuzz import fuzz, process

        all_plugins = []
        for library in self._libraries.values():
            all_plugins.extend(library.list_plugins())

        plugin_identifiers = [
            f"{i}:{p.identifier} {p.name}" for i, p in enumerate(all_plugins)
        ]

        matched_plugins = []
        results = process.extract(
            query,
            plugin_identifiers,
            scorer=fuzz.WRatio,
            limit=limit,
        )

        for match, score, _ in results:
            if score > threshold:  # Threshold for match quality
                # get the index from the match string
                index = int(match.split(":")[0])
                matched_plugins.append(all_plugins[index])

        return matched_plugins

    def get_plugins_by_type(
        self, plugin_type: Literal["spectral", "spectrum", "onset", "tempo"]
    ) -> List[PluginInfo]:
        """Get plugins filtered by type."""
        if not self._libraries:
            self.discover_plugins()

        filtered_plugins = []
        for library in self._libraries.values():
            for plugin_info in library.list_plugins():
                if plugin_type in plugin_info.identifier.lower():
                    filtered_plugins.append(plugin_info)
        return filtered_plugins

    def _load_plugin_from_cache(
        self, plugin_identifier: str, sample_rate: float
    ) -> Optional[VampPlugin]:
        """Try to load a plugin using cached library information."""
        cached_data = self._load_plugin_cache()
        if not cached_data:
            return None

        # Search cache for the plugin
        for lib_path_str, plugin_data_list in cached_data.items():
            for plugin_data in plugin_data_list:
                if plugin_data["identifier"] == plugin_identifier:
                    # Found it! Load only this library
                    lib_path = Path(lib_path_str)
                    if lib_path.exists():
                        library = self.host.load_library(str(lib_path))
                        if library:
                            self._libraries[lib_path_str] = library
                            plugin = library.instantiate_plugin(
                                plugin_data["index"], sample_rate
                            )
                            if plugin:
                                # Create a minimal PluginInfo for caching
                                plugin_info = library.list_plugins()[
                                    plugin_data["index"]
                                ]
                                return plugin, plugin_info
        return None

    def load_plugin(
        self,
        plugin_identifier: str,
        sample_rate: float = 44100.0,
        parameters: Optional[Dict[str, float]] = None,
    ) -> Optional[VampPlugin]:
        """Load a plugin by its identifier and optionally set parameters."""
        # Try cache-based loading first (much faster)
        cached_result = self._load_plugin_from_cache(plugin_identifier, sample_rate)
        if cached_result:
            plugin, plugin_info = cached_result
            # Set parameters if provided
            if parameters:
                self.set_plugin_parameters(plugin, parameters)
            self._plugins[plugin_identifier] = (plugin, plugin_info)
            return plugin

        # Fallback: search libraries one by one instead of full discovery
        library_paths = self.host.find_plugin_libraries()
        for lib_path in library_paths:
            library = self.host.load_library(lib_path)
            if library:
                self._libraries[lib_path] = library
                for plugin_info in library.list_plugins():
                    if plugin_info.identifier == plugin_identifier:
                        plugin = library.instantiate_plugin(
                            plugin_info.index, sample_rate
                        )
                        if plugin:
                            # Set parameters if provided
                            if parameters:
                                self.set_plugin_parameters(plugin, parameters)
                            self._plugins[plugin_identifier] = (plugin, plugin_info)
                            return plugin

        raise ValueError(f"Plugin '{plugin_identifier}' not found")

    def get_plugin_info(
        self, plugin_identifier: str, sample_rate: float = 44100.0
    ) -> Dict[str, Any]:
        """Get detailed information about a plugin."""
        # Load plugin if not already loaded
        if plugin_identifier not in self._plugins:
            plugin = self.load_plugin(plugin_identifier, sample_rate)
            if not plugin:
                raise ValueError(f"Plugin '{plugin_identifier}' not found")
        else:
            plugin, _ = self._plugins[plugin_identifier]

        return plugin.get_plugin_info()

    def get_plugin_outputs(
        self, plugin_identifier: str, sample_rate: float = 44100.0
    ) -> List[Dict[str, Any]]:
        """Get output descriptors for a plugin."""
        # Load plugin if not already loaded
        if plugin_identifier not in self._plugins:
            plugin = self.load_plugin(plugin_identifier, sample_rate)
            if not plugin:
                raise ValueError(f"Plugin '{plugin_identifier}' not found")
        else:
            plugin, _ = self._plugins[plugin_identifier]

        return list(plugin.get_output_descriptors())

    def get_plugin_parameters(
        self, plugin_identifier: str, sample_rate: float = 44100.0
    ) -> List[Dict[str, Any]]:
        """Get parameter descriptors for a plugin."""
        # Load plugin if not already loaded
        if plugin_identifier not in self._plugins:
            plugin = self.load_plugin(plugin_identifier, sample_rate)
            if not plugin:
                raise ValueError(f"Plugin '{plugin_identifier}' not found")
        else:
            plugin, _ = self._plugins[plugin_identifier]

        return list(plugin.get_parameter_descriptors())

    def list_plugin_outputs(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available plugins and their outputs."""
        if not self._libraries:
            self.discover_plugins()

        plugin_outputs = {}
        for lib_path, library in self._libraries.items():
            for plugin_info in library.list_plugins():
                try:
                    outputs = self.get_plugin_outputs(plugin_info.identifier)
                    plugin_outputs[plugin_info.identifier] = outputs
                except Exception:
                    # Skip plugins that can't be loaded
                    pass

        return plugin_outputs

    def list_plugin_parameters(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available plugins and their parameters."""
        if not self._libraries:
            self.discover_plugins()

        plugin_parameters = {}
        for lib_path, library in self._libraries.items():
            for plugin_info in library.list_plugins():
                try:
                    parameters = self.get_plugin_parameters(plugin_info.identifier)
                    plugin_parameters[plugin_info.identifier] = parameters
                except Exception:
                    # Skip plugins that can't be loaded
                    pass

        return plugin_parameters

    def set_plugin_parameters(
        self, plugin: VampPlugin, parameters: Dict[str, float]
    ) -> None:
        """Set parameters on a plugin instance."""
        for param_name, param_value in parameters.items():
            try:
                plugin.set_parameter_by_name(param_name, param_value)
            except Exception as e:
                print(f"Warning: Failed to set parameter '{param_name}': {e}")

    def get_plugin_parameters_dict(self, plugin: VampPlugin) -> Dict[str, float]:
        """Get current parameter values as a dictionary."""
        param_descriptors = plugin.get_parameter_descriptors()
        parameters = {}

        for param_desc in param_descriptors:
            param_id = param_desc["identifier"]
            try:
                value = plugin.get_parameter_by_name(param_id)
                if value is not None:
                    parameters[param_id] = value
            except Exception:
                # Skip parameters that can't be read
                pass

        return parameters

    def process_audio(
        self,
        plugin_identifier: str,
        audio_data: "np.ndarray",
        sample_rate: float = 44100.0,
        output_index: Optional[int] = None,
        parameters: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Process audio data using the rust logic with zero-copy NumPy."""
        import numpy as np

        # Load plugin if not already loaded
        if plugin_identifier not in self._plugins:
            plugin = self.load_plugin(plugin_identifier, sample_rate, parameters)
            if not plugin:
                raise ValueError(f"Plugin '{plugin_identifier}' not found")
        else:
            plugin, _ = self._plugins[plugin_identifier]
            # Set parameters if provided
            if parameters:
                self.set_plugin_parameters(plugin, parameters)

        # Prepare audio data for zero-copy processing
        if audio_data.ndim == 1:
            # Already mono - ensure it's float32 and C-contiguous
            audio_optimized = np.asarray(audio_data, dtype=np.float32, order="C")
            channels = 1
        elif audio_data.ndim == 2:
            # Handle 2D arrays - determine if it's (samples, channels) or (channels, samples)
            if audio_data.shape[1] > audio_data.shape[0]:
                # Transpose if it looks like (channels, samples) to get (samples, channels)
                audio_data = audio_data.T

            # Ensure it's float32 and C-contiguous
            audio_optimized = np.asarray(audio_data, dtype=np.float32, order="C")
            channels = audio_optimized.shape[1]
        else:
            raise ValueError("Audio data must be 1D or 2D")

        # Use the new zero-copy process_audio_nd method
        result = plugin.process_audio_nd(
            audio_optimized, sample_rate, channels, output_index
        )

        if result:
            return list(result)
        else:
            return []


def load_audio(
    file_path: Union[str, Path], sample_rate: Optional[float] = None
) -> tuple["np.ndarray", float]:
    """Load audio file using soundfile."""
    import soundfile as sf

    audio, sr = sf.read(str(file_path))
    if sample_rate is not None and sr != sample_rate:
        from soxr import resample

        audio = resample(x=audio, in_rate=sr, out_rate=sample_rate)
        sr = sample_rate
    return audio, sr


def process_audio_file(
    file_path: Union[str, Path],
    plugin_identifier: str,
    sample_rate: Optional[float] = None,
    parameters: Optional[Dict[str, float]] = None,
    use_zero_copy: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Process an audio file through a Vamp plugin."""

    # Load audio
    audio, sr = load_audio(file_path, sample_rate)
    processor = AudioProcessor()

    if use_zero_copy:
        # Use the new zero-copy NumPy method
        return processor.process_audio_rust(
            plugin_identifier, audio, sr, parameters=parameters, **kwargs
        )
    else:
        # Use the original Python method
        return processor.process_audio(
            plugin_identifier, audio, sr, parameters=parameters, **kwargs
        )
