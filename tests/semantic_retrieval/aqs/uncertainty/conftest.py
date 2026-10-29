import pytest


@pytest.fixture
def fusion_factory(tiny_clip):
    def build(*,aqs=None,fusion=None,mapping='a'*64):
        from semantic_retrieval.aqs.config import AqsConfig
        from semantic_retrieval.aqs.fusion_config import FusionConfig
        from semantic_retrieval.aqs.fusion_model import FusionAqsModel
        return FusionAqsModel(tiny_clip.config,tiny_clip.state_dict(),aqs=aqs or AqsConfig(True,2,.2),
            fusion=fusion or FusionConfig(True,.1),id_mapping_sha256=mapping,temporal_layers=1)
    return build
