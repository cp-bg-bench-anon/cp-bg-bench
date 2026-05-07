from __future__ import annotations

from omegaconf import OmegaConf

# Register the `${eval:...}` resolver used in scheduler configs.
OmegaConf.register_new_resolver("eval", lambda s: eval(s, {}), replace=True)  # noqa: S307
