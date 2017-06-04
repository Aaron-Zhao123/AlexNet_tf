from tensorpack import *
PATH = '/local/scratch/share/ImageNet/ILSVRC/ilsvrc12_test_lmdb'
ds = LMDBData(PATH, shuffle=False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start_test()
