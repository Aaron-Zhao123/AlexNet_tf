from tensorpack import *
class RawILSVRC12(DataFlow):
    def __init__(self):
        meta = dataset.ILSVRCMeta()
        self.imglist = meta.get_image_list('train')
        # we apply a global shuffling here because later we'll only use local shuffling
        np.random.shuffle(self.imglist)
        self.dir = os.path.join('/path/to/ILSVRC', 'train')
    def get_data(self):
        for fname, label in self.imglist:
            fname = os.path.join(self.dir, fname)
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]
    def size(self):
        return len(self.imglist)
ds0 = RawILSVRC12()
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, '/local/scratch/yaz21/ILSVRC-train.lmdb')
