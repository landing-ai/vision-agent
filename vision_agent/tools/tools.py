class Classifier:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self: image: Union[str, Image]) -> List[Dict]:
        raise NotImplementedError


class Detector:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self: image: Union[str, Image]) -> List[Dict]:
        raise NotImplementedError


class Segmentor:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self: image: Union[str, Image]) -> List[Dict]:
        raise NotImplementedError
