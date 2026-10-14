"""Sampler interfaces owned by ARFT, independent of unpublished verl modules."""
from abc import ABC, abstractmethod
from torch.utils.data import Sampler


class AbstractSampler(Sampler, ABC):
    pass


class AbstractCurriculumSampler(AbstractSampler):
    @abstractmethod
    def update(self, batch):
        """Advance the curriculum after an optimization batch."""
