import random


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buf = []
        self.ptr = 0

    def add(self, tr):
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.ptr] = tr
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)
