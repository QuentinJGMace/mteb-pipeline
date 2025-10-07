from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.evaluation.evaluators.Image.Any2AnyRetrievalEvaluator import (
    Any2AnyDenseRetrievalExactSearch,
)
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
)

logger = logging.getLogger(__name__)


class PipelineColPaliEngineEmbedder:
    """Base wrapper for `colpali_engine` models. Adapted from https://github.com/illuin-tech/colpali/tree/bebcdd6715dba42624acd8d7f7222a16a5daf848/colpali_engine/models"""

    def __init__(
        self,
        model_name: str,
        model_class: type,
        processor_class: type,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.mdl = model_class.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.mdl.eval()

        # Load processor
        self.processor = processor_class.from_pretrained(model_name)

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        import torchvision.transforms.functional as F

        all_embeds = []

        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(iterator):
                # batch may be list of tensors or PIL
                imgs = [
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                    for b in batch
                ]
                inputs = self.processor.process_images(imgs).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in texts[i : i + batch_size]
                ]
                inputs = self.processor.process_texts(batch).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings).T
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        return self.processor.score(a, b, device=self.device)


class PipelineColPaliWrapper:
    """Wrapper for ColPali models."""

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColPali, ColPaliProcessor

        model = PipelineColPaliEngineEmbedder(
            model_name=model_name,
            model_class=ColPali,
            processor_class=ColPaliProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )
        self.retriever = Any2AnyDenseRetrievalExactSearch(
            model, encode_kwargs={"batch_size": 16}
        )

    def pipeline_search(
        self,
        corpus,
        queries,
        task_name,
        top_k,
        return_sorted,
        **kwargs,
    ):
        return self.retriever.search(
            corpus,
            queries,
            task_name=task_name,
            top_k=top_k,
            score_function="cos_sim",
            return_sorted=return_sorted,
            **kwargs,
        )
