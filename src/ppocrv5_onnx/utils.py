from omegaconf import OmegaConf

# Load config and using via class

def load_config(config_path):
    return OmegaConf.load(config_path)
