from abc import ABCMeta, abstractmethod
from enum import Enum


class Strategy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def decide(self, predicted_movement):
        raise NotImplementedError


class SimpleStrategy(Strategy):
    def __init__(self):
        self.position = 'not_owned'

    def __str__(self):
        return 'SimpleStrategy'

    def decide(self, predicted_movement):
        if predicted_movement == 1:
            if self.position == 'not_owned':
                self.position = 'owned'
                return 'buy'
            elif self.position == 'owned':
                return 'hold'
        elif predicted_movement == 0:
            if self.position == 'not_owned':
                return 'wait'
            elif self.position == 'owned':
                self.position = 'not_owned'
                return 'sell'
        else:
            raise ValueError


