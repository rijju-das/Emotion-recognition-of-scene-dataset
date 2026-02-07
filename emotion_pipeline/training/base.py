from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def evaluate(self):
        ...
