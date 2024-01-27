from typing import Dict

class BaseMetric(object):
    def __call__(self, output, target) -> Dict[str, float]:
        raise NotImplementedError()