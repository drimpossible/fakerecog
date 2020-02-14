from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class VideoReaderPipeline(Pipeline):
    def __init__(self, filename, batch_size, sequence_length, num_threads, device_id):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.reader = ops.VideoReader(device="gpu", filenames=filename, sequence_length=sequence_length, normalized=False, image_type=types.RGB, dtype=types.FLOAT)

    def define_graph(self):
        output = self.reader(name="Reader")
        return output

class VideoLoader():
    def __init__(self, filename, batch_size, sequence_length, workers, device_id):
        self.pipeline = VideoReaderPipeline(filename=filename, batch_size=batch_size, sequence_length=sequence_length, num_threads=workers, device_id=device_id)
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         self.epoch_size,
                                                         auto_reset=True, fill_last_batch=False)
    def __len__(self):
        return int(self.epoch_size)
    def __iter__(self):
        return self.dali_iterator.__iter__()