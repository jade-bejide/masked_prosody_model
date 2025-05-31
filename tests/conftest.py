import pytest
import torch
import os
import subprocess
import sys
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "torch_version(version): mark test to run with specific PyTorch version"
    )

@pytest.fixture(scope="session")
def torch_version():
    """Get the current PyTorch version."""
    return torch.__version__

@pytest.fixture(scope="session")
def is_torch_2_0_or_higher():
    """Check if PyTorch version is 2.0 or higher."""
    return int(torch.__version__.split('.')[0]) >= 2

def pytest_collection_modifyitems(config, items):
    """Modify test items based on PyTorch version."""
    torch_version = torch.__version__
    is_torch_2_0_or_higher = int(torch_version.split('.')[0]) >= 2

    for item in items:
        # Skip tests marked with torch_version if version doesn't match
        if hasattr(item, 'callspec') and 'torch_version' in item.callspec.params:
            required_version = item.callspec.params['torch_version']
            if required_version != torch_version:
                item.add_marker(pytest.mark.skip(
                    reason=f"Test requires PyTorch {required_version}, but running with {torch_version}"
                ))

        # Add version-specific markers
        if is_torch_2_0_or_higher:
            item.add_marker(pytest.mark.torch_2_0)
        else:
            item.add_marker(pytest.mark.torch_1_x)

@pytest.fixture(autouse=True)
def check_torch_version():
    """Check if PyTorch version is compatible with the test."""
    torch_version = torch.__version__
    major_version = int(torch_version.split('.')[0])
    
    if major_version < 1 or major_version > 2:
        pytest.skip(f"PyTorch version {torch_version} is not supported. Please use PyTorch 1.x or 2.x") 