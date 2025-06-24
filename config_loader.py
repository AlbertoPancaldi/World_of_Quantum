"""
Configuration Management for Quantum Layout Optimization

Loads and validates configuration settings from YAML files.
Provides easy access to all user-configurable parameters.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class ConfigLoader:
    """
    Configuration loader and manager.
    
    Loads YAML configuration files and provides structured access
    to all settings with validation and defaults.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        self.load_config()
        self.validate_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def validate_config(self) -> None:
        """Validate configuration structure and values."""
        required_sections = [
            'experiment', 'backend', 'benchmarks', 'layout_optimization',
            'transpilation', 'targets'
        ]
        
        missing_sections = [sec for sec in required_sections if sec not in self.config]
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate backend
        valid_backends = ['FakeBrisbane', 'FakePerth', 'FakeSherbrooke']
        if self.config['backend']['name'] not in valid_backends:
            raise ValueError(f"Invalid backend: {self.config['backend']['name']}. "
                           f"Valid options: {valid_backends}")
        
        # Validate active suite
        valid_suites = ['quantum_volume', 'application_circuits', 'qasm_circuits']
        if self.config['benchmarks']['active_suite'] not in valid_suites:
            raise ValueError(f"Invalid active_suite: {self.config['benchmarks']['active_suite']}. "
                           f"Valid options: {valid_suites}")
    
    def get_circuit_sizes(self) -> List[int]:
        """Get list of circuit sizes to run."""
        return self.config['benchmarks']['circuit_sizes']
    
    def get_active_benchmarks(self) -> List[str]:
        """Get list of enabled benchmark suite names."""
        return [self.config['benchmarks']['active_suite']]
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return False  # Simplified config doesn't have demo mode
    
    def get_demo_sizes(self) -> List[int]:
        """Get circuit sizes for demo mode."""
        return self.get_circuit_sizes()
    
    def get_output_directory(self) -> Path:
        """Get output directory path."""
        output_dir = Path(self.config['experiment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    # Convenience methods for common parameter access
    def get_seed(self) -> int:
        """Get the global random seed."""
        return self.config['experiment']['seed']
    
    def get_backend_name(self) -> str:
        """Get the backend name."""
        return self.config['backend']['name']
    
    def get_active_suite(self) -> str:
        """Get the active benchmark suite."""
        return self.config['benchmarks']['active_suite']
    
    def get_optimization_level(self) -> int:
        """Get the baseline optimization level."""
        return self.config['transpilation']['baseline_optimization_level']
    
    def get_depth_factor(self) -> float:
        """Get quantum volume depth factor."""
        return self.config['benchmarks']['quantum_volume']['depth_factor']
    
    def get_annealing_max_time(self) -> float:
        """Get annealing maximum time."""
        return self.config['layout_optimization']['annealing']['max_time_seconds']
    
    def get_qaoa_layers(self) -> int:
        """Get QAOA layers parameter."""
        return self.config['benchmarks']['application_circuits']['qaoa']['layers']
    
    def get_vqe_reps(self) -> int:
        """Get VQE repetitions parameter."""
        return self.config['benchmarks']['application_circuits']['vqe']['ansatz_reps']
    
    def print_config_summary(self) -> None:
        """Print a summary of the loaded configuration."""
        print("🔧 Configuration Summary")
        print("=" * 40)
        
        exp = self.config['experiment']
        print(f"Experiment: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Seed: {exp['seed']}")
        
        backend = self.config['backend']
        print(f"\nBackend: {backend['name']} ({backend['num_qubits']} qubits)")
        
        print(f"\nBenchmark suite: {self.config['benchmarks']['active_suite']}")
        print(f"Circuit sizes: {self.get_circuit_sizes()}")
        
        layout = self.config['layout_optimization']
        print(f"\nLayout algorithm: {layout['algorithm']}")
        
        targets = self.config['targets']
        print(f"\nTargets:")
        print(f"  CX reduction: ≥{targets['cx_reduction_target_percent']}%")
    
    def save_runtime_config(self, filename: Optional[str] = None) -> Path:
        """
        Save current configuration to output directory.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved configuration file
        """
        if filename is None:
            filename = f"config_runtime_{self.config['experiment']['name']}.yaml"
        
        output_path = self.get_output_directory() / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        return output_path


# Global configuration instance
_config_instance = None


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Load global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance


def get_config() -> ConfigLoader:
    """
    Get global configuration instance.
    
    Returns:
        ConfigLoader instance
        
    Raises:
        RuntimeError: If configuration hasn't been loaded yet
    """
    global _config_instance
    if _config_instance is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        config.print_config_summary()
        print("\n✅ Configuration loaded successfully!")
        
        # Test structured access
        print(f"\nDemo mode: {config.is_demo_mode()}")
        print(f"Circuit sizes: {config.get_circuit_sizes()}")
        print(f"Active benchmarks: {config.get_active_benchmarks()}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
