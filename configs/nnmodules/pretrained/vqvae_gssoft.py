from configs.class_builder import ClassBuilder, ParamSlot
from configs.utils.pretrained_model_builder import PretrainedModelBuilder
from cbench.nn.models.vqvae_model_v2 import GSSOFT

config = PretrainedModelBuilder(GSSOFT,
    channels=ParamSlot("channels"),
    latent_dim=ParamSlot("latent_dim"),
    num_embeddings=ParamSlot("num_embeddings"),
    embedding_dim=ParamSlot("embedding_dim"),
    input_shift=ParamSlot("input_shift"),
    lr=ParamSlot("lr"),
)