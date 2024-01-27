from configs.class_builder import ClassBuilder, ParamSlot
from cbench.modules.prior_model.autoencoder_v2 import VQVAEV2BackboneAutoEncoderPriorModel

config = ClassBuilder(VQVAEV2BackboneAutoEncoderPriorModel,
    # latent_channels=ParamSlot(),
)