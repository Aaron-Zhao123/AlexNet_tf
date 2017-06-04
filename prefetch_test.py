from tensorpack import *
import lmdb
PATH = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC'
ds0 = dataset.ILSVRC12(PATH, 'train', shuffle = True)
ds1 = PrefetchDataZMQ(ds0, nr_proc=8)
dftools.dump_dataflow_to_lmdb(ds1, './ILSVRC-train.lmdb')
