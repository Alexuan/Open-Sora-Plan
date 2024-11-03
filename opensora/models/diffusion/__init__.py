
from .latte.modeling_latte import Latte_models
from .latte.modeling_latte_a2v import Latte_models_A2V

Diffusion_models = {}
Diffusion_models.update(Latte_models)
Diffusion_models.update(Latte_models_A2V)

    