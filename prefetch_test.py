from tensorpack import *
import lmdb
PATH = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC/'
CONVERT_LMDB = 1

if CONVERT_LMDB:
    ds0 = dataset.ILSVRC12(PATH, 'train', shuffle = True)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, './ILSVRC-train.lmdb')
    ds2 = dataset.ILSVRC12(PATH, 'val', shuffle = True)
    ds3 = PrefetchDataZMQ(ds2, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds3, './ILSVRC-val.lmdb')

# check the training data
db_dir = '/local/scratch/share/ImageNet/tensorflow/'
ds = LMDBData(db_dir + 'ILSVRC-train.lmdb', shuffle=False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start_test()
