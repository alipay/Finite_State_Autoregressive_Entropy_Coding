from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder_v2 import VAELosslessAutoEncoderPriorModel

config = ClassBuilder(VAELosslessAutoEncoderPriorModel,
    # latent_channels=ParamSlot(),
)