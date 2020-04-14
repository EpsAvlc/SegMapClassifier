import numpy as np

def to_onehot(y, n_classes):
    y_onehot = np.zeros((len(y), n_classes))
    for i, cls in enumerate(y):
        y_onehot[i, cls] = 1

    return y_onehot

class Generator(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        n_classes,
        train=True,
        batch_size=16,
        shuffle=False,
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_segments = len(self.segment_ids)
        self.n_batches = int(np.ceil(float(self.n_segments) / batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.segment_ids)

        self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]
#         print(self.segment_ids)
        self._i = self._i + self.batch_size
        if self._i >= self.n_segments:
            self._i = 0

        batch_segments, batch_classes = self.preprocessor.get_processed(
            self.batch_ids, train=self.train
        )

        batch_segments = batch_segments[:, np.newaxis, :, :, :]
        # batch_segments = np.insert(batch_segments, 1, 1, axis = 1)
        # batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes