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

class DALILoader():
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


from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os


def get_all_image_paths(root_dir):
    image_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        image_paths += [dirpath+'/'+f for f in filenames if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
    return image_paths


class SimpleFolderLoader(VisionDataset):
    """A generic data loader where the samples are arranged in the following way:

        root/folder_1/xxx.ext
        root/folder_1/xxy.ext
        .
        .
        .
        root/folder_n/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, transform=None):
        super(SimpleFolderLoader, self).__init__(root, transform=transform)
        image_paths = get_all_image_paths(self.root)
        if len(image_paths) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

        self.loader = loader
        self.image_paths = image_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_path, image).
        """
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return path, sample

    def __len__(self):
        return len(self.image_paths)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class BurstLoader(SimpleFolderLoader):
    """Fast batched dataloader for preprocessing of bursted datasets"""

    def __init__(self, root, transform=None, loader=default_loader):
        super(BurstLoader, self).__init__(root, loader, transform=transform)
        self.img_paths = self.image_paths

def dfdc_burst_loader(root):
    loader = BurstLoader(root,
        transform=transforms.ToTensor()




import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
      
      input:
          x: the input signal 
          window_len: the dimension of the smoothing window; should be an odd integer
          window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
              flat window will produce a moving average smoothing.
  
      output:
          the smoothed signal
          
      example:
  
      t=linspace(-2,2,0.1)
      x=sin(t)+randn(len(t))*0.1
      y=smooth(x)
      
      see also: 
      
      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
      scipy.signal.lfilter
   
      TODO: the window parameter could be the window itself if an array instead of a string
      NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
      """ 
       
      if x.ndim != 1:
          raise ValueError, "smooth only accepts 1 dimension arrays."
  
      if x.size < window_len:
          raise ValueError, "Input vector needs to be bigger than window size."
          
  
      if window_len<3:
          return x
      
      
      if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
          raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
      
  
      s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
      #print(len(s))
      if window == 'flat': #moving average
          w=numpy.ones(window_len,'d')
      else:
          w=eval('numpy.'+window+'(window_len)')
      
      y=numpy.convolve(w/w.sum(),s,mode='valid')
       return y  




















