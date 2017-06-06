'''
The code here takes the following link as a reference
http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
'''
from tensorpack import *
import lmdb
import numpy as np
import os
import sys
PATH = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC'
CONVERT_LMDB_TRAIN = 0
CONVERT_LMDB_VAL = 0
EFFICIENT_FLOW = 0

if CONVERT_LMDB_TRAIN:
    class RawILSVRC12(DataFlow):
        def __init__(self):
            meta = dataset.ILSVRCMeta()
            self.imglist = meta.get_image_list('train')
            # we apply a global shuffling here because later we'll only use local shuffling
            np.random.shuffle(self.imglist)
            self.dir = os.path.join(PATH, 'train')
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
    dftools.dump_dataflow_to_lmdb(ds1, './ILSVRC-train.lmdb')

if CONVERT_LMDB_VAL:
    class RawILSVRC12(DataFlow):
        def __init__(self):
            meta = dataset.ILSVRCMeta()
            self.imglist = meta.get_image_list('val')
            # we apply a global shuffling here because later we'll only use local shuffling
            np.random.shuffle(self.imglist)
            self.dir = os.path.join(PATH, 'val')
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
    dftools.dump_dataflow_to_lmdb(ds1, './ILSVRC-val.lmdb')
    sys.exit()

# check the training data
if EFFICIENT_FLOW:
    db_dir = './'
    fix_path = os.path.join(db_dir, 'ILSVRC-train.lmdb')
    ds = LMDBData(fix_path, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = PrefetchData(ds, 5000, 1)
    # ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    # ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = PrefetchDataZMQ(ds, 25)
    ds = BatchData(ds, 256)
else:
    db_dir = './'
    ds = LMDBData(db_dir + 'ILSVRC-train.lmdb', shuffle=False)
    ds = BatchData(ds, 256, use_list=True)

TestDataSpeed(ds).start_test()
