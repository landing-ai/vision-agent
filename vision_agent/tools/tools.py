from typing import Dict, List, Union

from PIL.Image import Image as ImageType


class Classifier:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError


class Detector:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError


class Segmentor:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError
