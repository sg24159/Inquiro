from pathlib import Path

import yaml

_agents_path = Path(__file__).parent / "agents.yaml"
with open(_agents_path) as f:
    agents_config = yaml.safe_load(f)

