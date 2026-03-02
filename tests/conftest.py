import os

import pytest

# Set dummy env vars so Settings() can instantiate during test imports.
# Tests that need real API access should use @pytest.mark.integration.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
