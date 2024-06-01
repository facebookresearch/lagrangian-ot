# Copyright (c) Meta Platforms, Inc. and affiliates

class EMAMeter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
        self.last_value = None

    def update(self, value):
        if self.value is None:
            self.value = value
            self.last_value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
            self.last_value = value
