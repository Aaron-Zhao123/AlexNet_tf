from tensorpack import *
PATH = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC'
ds = dataset.ILSVRC12(PATH, 'train', shuffle = False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start_test()
