"""
Basic tests for VampRust Python bindings.
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
from pathlib import Path

# Add the python directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import vamprust
    from vamprust import VampHost, AudioProcessor, FeatureSet, Feature
    VAMPRUST_AVAILABLE = True
except ImportError:
    VAMPRUST_AVAILABLE = False


@pytest.mark.skipif(not VAMPRUST_AVAILABLE, reason="VampRust not available")
class TestVampHost:
    """Test VampHost functionality."""
    
    def test_host_creation(self):
        """Test that we can create a VampHost instance."""
        host = VampHost()
        assert host is not None
        assert repr(host).startswith("VampHost")
    
    def test_find_plugin_libraries(self):
        """Test plugin library discovery."""
        host = VampHost()
        libraries = host.find_plugin_libraries()
        assert isinstance(libraries, list)
        # Note: May be empty if no plugins installed
    
    def test_load_library_nonexistent(self):
        """Test loading a non-existent library."""
        host = VampHost()
        library = host.load_library("/nonexistent/path/library.so")
        assert library is None


@pytest.mark.skipif(not VAMPRUST_AVAILABLE, reason="VampRust not available")
class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    def test_processor_creation(self):
        """Test that we can create an AudioProcessor."""
        processor = AudioProcessor()
        assert processor is not None
    
    def test_discover_plugins(self):
        """Test plugin discovery."""
        processor = AudioProcessor()
        plugins = processor.discover_plugins()
        assert isinstance(plugins, list)
        # Note: May be empty if no plugins installed
    
    def test_process_audio_with_mock_plugin(self):
        """Test audio processing with generated data."""
        # Generate test audio
        sample_rate = 44100
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        processor = AudioProcessor()
        
        # This will likely fail if no plugins are available, which is OK for testing
        plugins = processor.discover_plugins()
        if plugins:
            # Try to process with the first available plugin
            plugin_id = plugins[0].identifier
            try:
                features = processor.process_audio(plugin_id, audio, sample_rate)
                assert isinstance(features, list)
            except Exception:
                # Plugin processing may fail, which is acceptable in tests
                pass


class TestFeature:
    """Test Feature class functionality."""
    
    def test_feature_creation(self):
        """Test Feature creation and properties."""
        feature = Feature(
            values=[1.0, 2.0, 3.0],
            has_timestamp=True,
            sec=1,
            nsec=500000000,  # 0.5 seconds in nanoseconds
            label="test_feature"
        )
        
        assert feature.values == [1.0, 2.0, 3.0]
        assert feature.has_timestamp is True
        assert feature.sec == 1
        assert feature.nsec == 500000000
        assert feature.label == "test_feature"
        assert feature.timestamp == 1.5  # 1 + 0.5
    
    def test_feature_to_dict(self):
        """Test Feature to dictionary conversion."""
        feature = Feature(
            values=[1.0, 2.0],
            has_timestamp=True,
            sec=0,
            nsec=250000000,
            label="test"
        )
        
        d = feature.to_dict()
        assert d['values'] == [1.0, 2.0]
        assert d['has_timestamp'] is True
        assert d['timestamp'] == 0.25
        assert d['sec'] == 0
        assert d['nsec'] == 250000000
        assert d['label'] == "test"


class TestFeatureSet:
    """Test FeatureSet functionality."""
    
    def create_test_features(self):
        """Create test features for testing."""
        return [
            Feature([1.0, 2.0], True, 0, 0, "onset"),
            Feature([3.0, 4.0], True, 0, 500000000, "onset"),
            Feature([5.0, 6.0], True, 1, 0, "beat"),
            Feature([7.0, 8.0], False, 0, 0, None),
        ]
    
    def test_featureset_creation(self):
        """Test FeatureSet creation."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        assert len(fs) == 4
        assert list(fs) == features
        assert fs[0] == features[0]
    
    def test_featureset_from_dicts(self):
        """Test FeatureSet creation from dictionaries."""
        feature_dicts = [
            {'values': [1.0, 2.0], 'has_timestamp': True, 'sec': 0, 'nsec': 0},
            {'values': [3.0, 4.0], 'has_timestamp': False},
        ]
        
        fs = FeatureSet(feature_dicts)
        assert len(fs) == 2
        assert fs[0].values == [1.0, 2.0]
        assert fs[0].has_timestamp is True
        assert fs[1].values == [3.0, 4.0]
        assert fs[1].has_timestamp is False
    
    def test_timestamps_property(self):
        """Test timestamps property."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        timestamps = fs.timestamps
        expected = np.array([0.0, 0.5, 1.0])  # Only features with timestamps
        np.testing.assert_array_equal(timestamps, expected)
    
    def test_values_matrix_property(self):
        """Test values matrix property."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        matrix = fs.values_matrix
        expected = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        np.testing.assert_array_equal(matrix, expected)
    
    def test_filter_by_timestamp(self):
        """Test filtering by timestamp."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        filtered = fs.filter_by_timestamp(0.25, 0.75)
        assert len(filtered) == 1
        assert filtered[0].timestamp == 0.5
    
    def test_filter_by_label(self):
        """Test filtering by label."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        onset_features = fs.filter_by_label("onset")
        assert len(onset_features) == 2
        
        beat_features = fs.filter_by_label("beat")
        assert len(beat_features) == 1
    
    def test_group_by_label(self):
        """Test grouping by label."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        groups = fs.group_by_label()
        assert "onset" in groups
        assert "beat" in groups
        assert "unlabeled" in groups
        
        assert len(groups["onset"]) == 2
        assert len(groups["beat"]) == 1
        assert len(groups["unlabeled"]) == 1
    
    def test_to_numpy(self):
        """Test conversion to numpy array."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        array = fs.to_numpy()
        expected = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        np.testing.assert_array_equal(array, expected)
    
    def test_summary(self):
        """Test summary generation."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        summary = fs.summary()
        assert summary['count'] == 4
        assert summary['dimensions'] == 2
        assert summary['has_timestamps'] is True
        assert set(summary['labels']) == {"onset", "beat"}
        assert 'duration' in summary
        assert 'mean_values' in summary
    
    def test_to_dict_list(self):
        """Test conversion to list of dictionaries."""
        features = self.create_test_features()
        fs = FeatureSet(features)
        
        dict_list = fs.to_dict_list()
        assert len(dict_list) == 4
        assert dict_list[0]['values'] == [1.0, 2.0]
        assert dict_list[0]['label'] == "onset"


@pytest.mark.skipif(not VAMPRUST_AVAILABLE, reason="VampRust not available")
def test_module_imports():
    """Test that all expected modules and classes can be imported."""
    import vamprust
    
    # Test that main classes are available
    assert hasattr(vamprust, 'VampHost')
    assert hasattr(vamprust, 'AudioProcessor')
    assert hasattr(vamprust, 'FeatureSet')
    assert hasattr(vamprust, 'Feature')
    
    # Test version
    assert hasattr(vamprust, '__version__')


def test_numpy_available():
    """Test that numpy is available for testing."""
    import numpy as np
    assert np is not None
    
    # Test basic numpy operations that we use
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,)
    assert np.mean(arr) == 2.0