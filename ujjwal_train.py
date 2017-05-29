import os
import tensorflow as tf
import threading
import time
import numpy as np
from Read_labelclsloc import readlabel
from methods import *
from skimage import io, img_as_float

epoch = 0
finished = False
old_epoch = 0

def ReadTrainMNIST(traindir):
    categories = []
    trainimgs = []
    trainlbls = []
    for entry in os.scandir(traindir):
        if entry.is_dir():
            categories.append(entry.name)
            for files in os.scandir(os.path.join(traindir, entry.name)):
                if files.is_file() and files.name.endswith('.png'):
                    trainimgs.append(os.path.join(traindir, entry.name,
                                                  files.name))
                    trainlbls.append(int(entry.name))

    return trainimgs, trainlbls



def ReadTrain(traindir):
    """
    Reads the training subset of CLS-LOC challenge of ILSVRC 2015
    :param traindir: The folder containing the training class subfolders.
    :return: A list (trainimgs) of all training images, a dictionary (
    classdict) mapping the class
    names to positive integers and a list (trainlbls) of the same length as
    trainimgs, containing corresponding positive integers for the
    corresponding classes of the images.
    """
    categories = []
    timeinit = time.time()
    trainimgs = []
    trainlbls = []
    for entry in os.scandir(traindir):
        if entry.is_dir():
            categories.append(entry.name)
            for files in os.scandir(os.path.join(traindir, entry.name)):
                if files.is_file() and files.name.endswith('.JPEG'):
                    trainimgs.append(os.path.join(traindir, entry.name
                                                  , files.name))
    classdict = dict(zip(categories, range(len(categories))))
    print("""Time taken to identify the training images and prepare the class
    label dictionary = %.2f seconds""" % (time.time() - timeinit))
    timeinit = time.time()
    counter = 0
    for img in trainimgs:
        lbl = classdict[os.path.basename(os.path.dirname(img))]
        trainimgs[counter] = trainimgs[counter]
        trainlbls.append(lbl)
        counter += 1
    print("""Time taken to identify the labels for training images = %.2f
    seconds""" % (time.time() - timeinit))
    return trainimgs, trainlbls, classdict


def ReadVal(valdir, classdict):
    """
    Reads the validation images of the CLS-LOC challenge of the ILSVRC 2015
    :param valdir: the folder containing the validation subset images.
    :param classdict: Dictionary mapping ILSVRC training labels to positive
    integers, computed using `ReadTrain()`
    :return: A dictionary. Keys are the full paths to the validation images.
    The values are the corresponding labels.
    """
    anndir = os.path.join(valdir, '..', '..', '..',
                          'Annotations', 'CLS-LOC',
                          'val')
    valdict = {}
    timeinit = time.time()
    for entry in os.scandir(valdir):
        valfile = os.path.join(valdir, entry.name)
        annfile = os.path.join(anndir,
                               os.path.splitext(os.path.basename(entry.name))[
                                   0]
                               + '.xml')
        labels = readlabel(annfile)
        labels = list(set(labels))
        lbl = map(lambda x: classdict[x], labels)
        lbl = list(lbl)
        valdict[valfile] = lbl
        if len(lbl) > 1:
            print('Multiple labels found.')
            print(lbl)
    print("""Time taken to identify the labels for validation image = %.2f
         seconds""" % (time.time() - timeinit))
    return valdict


def ReadTest(testdir):
    testimgs = []
    for entry in os.scandir(testdir):
        if entry.is_file():
            testimgs.append(os.path.join(testdir, entry.name))
    return testimgs

def PreProcessing(example):
    """
    Preprocesses an image to produce augmented versions of it.
    For each image, it is resized such that the aspect ratio is preserved
    and the minimum dimension of the resized image is 256 pixels. Then
    5 random crops of size (221, 221) are taken from the image including
    their flipped versions. This results in a total of 10 augmented samples.
    They are then packed into an array and returned back.
    :param example: Name of the image file to be processed
    :return: A numpy array containing 10 stacks of augmented samples.
    """
    time_init = time.time()
    image = io.imread(example)
    imagenet_mean = np.array([104, 117, 123], dtype=np.float)

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image = image.astype(np.float)
    image-=imagenet_mean
    image = resize_min(image, 256)

    tl = crop_random_tl(image, (221, 221))
    tr = crop_random_tr(image, (221, 221))
    bl = crop_random_bl(image, (221, 221))
    br = crop_random_br(image, (221, 221))

    cent = crop_center(image, (221, 221))

    tl_ref = flip_patch(tl)
    tr_ref = flip_patch(tr)
    bl_ref = flip_patch(bl)
    br_ref = flip_patch(br)
    cent_ref = flip_patch(cent)
    """
    return np.asarray([cent])
    """
    return np.asarray([tl, tr, bl, br, cent, tl_ref, tr_ref, bl_ref,
    br_ref, cent_ref])





def traininginfo( traindir, valdir, testdir):
    """
    Saves the list of training images, their labels, a dictionary mapping
    CLS-LOC class names to positive integers and a dictionary containing
    validation data.
    :return: None
    """

    trainimgs, trainlbls, classdict = ReadTrain(traindir)
    valdict = ReadVal(valdir, classdict)
    testimgs = ReadTest(testdir)

    """
    trainimgs, trainlbls = ReadTrainMNIST(traindir)
    return trainimgs, trainlbls
    """
    return trainimgs, trainlbls, classdict, valdict, testimgs

def EnqueueFileNames( filename_enq, graph):
    """
    Enqueues the list of images and labels into the self.filename_queue
    :param sess: A TensorFlow session in which the operation of
    enqueueing has to be performed
    :return: None
    """
    sess = tf.Session(graph=graph)
    sess.run(filename_enq)
    return None

def EnqueueData(filename_queue_size, filename_enq, filename_queue_close, filename_dq, data_enq, image_placeholder, label_placeholder,  sess, coord, lock, maxepochs, graph):
    """
    Dequeues an image name and its label from self.filename_queue and
    preprocesses it to produce 10 augmented samples to it and then
    enqueues them into the self.data_queue.
    :param sess: TensorFlow session in which the enqueueing has to be done
    :param coord: a tf.Coordinator() object
    :param lock: a threading.Lock() object for synchronizing multiple
    python threads running this function
    :param maxepochs: Maximum number of epochs of data which have to be
    preprocessed and enqueued
    :return: None
    """
    global epoch
    global finished
    global old_epoch
    while True:
        lock.acquire()
        if (sess.run(filename_queue_size) == 0) and (
                epoch < maxepochs):
            sess.run(filename_enq)
            #EnqueueFileNames(filename_enq, sess)
            old_epoch = epoch
            epoch += 1
        if (sess.run(filename_queue_size) == 0) and (
            epoch == maxepochs):
            if not coord.should_stop():
                coord.request_stop()
                sess.run(filename_queue_close)
                lock.release()
                finished = True
                break
        try:
            data = sess.run(filename_dq)
        except tf.errors.OutOfRangeError:
            lock.release()
            break
        lock.release()
        img = data["Filename"]
        img = img.decode(encoding='UTF-8')
        label = data["Label"]
        preprocessed = PreProcessing(img)
        #        print(preprocessed)
        label = [label] * 10
        sess.run(data_enq, feed_dict={
            image_placeholder: preprocessed,
            label_placeholder: label
        })
    return None



def Train(traindir, valdir, testdir, batchsize=128, numthreads=50, maxepochs=20):
    global epoch
    global old_epoch
    graph = tf.Graph()
    trainimgs, trainlbls, classdict, valdict, testimgs = traininginfo(traindir, valdir, testdir)
    #trainimgs, trainlbls = traininginfo(traindir, valdir, testdir)
    batch = batchsize
    with graph.as_default():
        trainimgs_tensor = tf.constant(trainimgs,
                                       dtype=tf.string)
        trainlbls_tensor = tf.constant(trainlbls, dtype=tf.int32)
        trainimgs_dict = {}
        trainimgs_dict["Filename"] = trainimgs_tensor
        trainimgs_dict["Label"] = trainlbls_tensor
        filename_queue = tf.RandomShuffleQueue(capacity=len(trainimgs),
                                               min_after_dequeue=0,
                                               dtypes=[tf.string, tf.int32],
                                               names=["Filename", "Label"],
                                               name="filename_queue")
        filename_queue_size = filename_queue.size()
        image_holder = tf.placeholder(dtype=tf.float32,
                                      shape=(None, 221, 221, 3),
                                      name="image_holder")
        label_holder = tf.placeholder(dtype=tf.int32,
                                      shape=(None),
                                      name="label_holder")
        data_queue = tf.RandomShuffleQueue(capacity=2048,
                                           min_after_dequeue=32,
                                           dtypes=[tf.float32,
                                                   tf.int32],
                                           shapes=[[221, 221, 3], []],
                                           name="data_queue")
        data_queue_size = data_queue.size()
        filename_enq = filename_queue.enqueue_many(
            trainimgs_dict)
        filename_dq = filename_queue.dequeue()
        filename_queue_close = filename_queue.close()
        data_enq = data_queue.enqueue_many([image_holder,
                                            label_holder])
        batchsize = tf.constant(batchsize, dtype=tf.int32)

        [batch_images, batch_labels] = data_queue.dequeue_up_to(batchsize)

        scale = tf.constant(0.00001, dtype=tf.float32)
        conv1 = tf.layers.conv2d(batch_images, filters=96,
                                     kernel_size=7,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.zeros_initializer(),
                                     kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv1')
        tf.summary.image("Training batch", batch_images, max_outputs=100)

        l1weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1') if v.name.endswith('kernel:0')]

        l1weights = tf.stack(l1weights)

        l1weights = tf.squeeze(l1weights, [0])

        wt_min = tf.reduce_min(l1weights)

        wt_max = tf.reduce_max(l1weights)

        l1weights = (l1weights - wt_min) / (wt_max - wt_min)

        l1weights = tf.transpose(l1weights, [3, 0, 1, 2])


        tf.summary.image('Layer1_Weights', l1weights, max_outputs=96)

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=3,
                                            name='pool1')
        conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=7,
                                     strides=1,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.5),
                                     kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2,
                                            name='pool2')
        conv3 = tf.layers.conv2d(pool2, filters=512, kernel_size=3,
                                     strides=1,
                                     activation=tf.nn.relu, padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.zeros_initializer(),
                                     kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv3')
        conv4 = tf.layers.conv2d(conv3, filters=512, kernel_size=3,
                                     strides=1,
                                     activation=tf.nn.relu, padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.5),
                                     kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv4')
        conv5 = tf.layers.conv2d(conv4, filters=1024, kernel_size=3,
                                     strides=1,
                                     activation=tf.nn.relu, padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.5),
                                     kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv5')
        conv6 = tf.layers.conv2d(conv5, filters=1024, kernel_size=3,
                                     strides=1,
                                     activation=tf.nn.relu, padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.5),
                                  kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=None,
                                     name='conv6')
        pool3 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=3,
                                            name='pool3')
        pool3_reshaped = tf.contrib.layers.flatten(pool3)
        fc1 = tf.layers.dense(pool3_reshaped, units=4096,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                  kernel_regularizer=tf.nn.l2_loss,
                                  bias_initializer=tf.ones_initializer(),
                                  bias_regularizer=None,
                                  name='fc1')
        drop1 = tf.layers.dropout(fc1, training=True,name='drop1')
        fc2 = tf.layers.dense(drop1, units=4096,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                  kernel_regularizer=tf.nn.l2_loss,
                                  bias_initializer=tf.ones_initializer(),
                                  bias_regularizer=None,
                                  name='fc2')
        drop2 = tf.layers.dropout(fc2, training=True, name='drop2')
        output = tf.layers.dense(drop2, units=1000, activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     bias_initializer=tf.zeros_initializer(),

                                     name='output')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output,
                labels=batch_labels,
                name="cross_entropy_per_example")
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                                name="mean_cross_entropy")





        reg = tf.multiply(scale, tf.add_n(tf.losses.get_regularization_losses()))

        totalloss = tf.add(cross_entropy_mean, reg,name="totalloss")
        tf.summary.scalar('Total_Loss', totalloss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(totalloss)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.6, use_nesterov=False).minimize(totalloss)
        correct_pred = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32),
                                batch_labels)

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('Accuracy', accuracy)

        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('./logs',
                                             sess.graph)
        coordinator = tf.train.Coordinator()
        lock = threading.Lock()
        sess.run(init_op)
        threads = [threading.Thread(target=EnqueueData, args=(
            filename_queue_size, filename_enq, filename_queue_close, filename_dq,
            data_enq, image_holder, label_holder, sess, coordinator, lock, maxepochs, graph),
                                    daemon=True) for i in range(numthreads)]

        #names = [n.name for n in graph.as_graph_def().node]
        #print(names)

        for t in threads:
            t.start()
        counter = 0
        summary_counter = 0
        myepoch = 0
        while not (sess.run(data_queue_size) == 0 and finished):
            [a, cost, acc, summaries, regul] = sess.run([optimizer, totalloss, accuracy, merged, reg])

            print(regul)
            counter+=1
            summary_counter+=1

            print("Current epoch = %d, Images processed = %d, Cost = %.4f, accuracy = %.4f"%(epoch, batch * counter, cost, acc))
            if (summary_counter == 100):
                print('Adding summaries')
                train_writer.add_summary(summaries, counter)
                summary_counter = 0
            if (myepoch != epoch):
                print('Now saving model for epoch number {}'.format(myepoch))
                savename = 'model-epoch' + str(myepoch) + '.tfmodel'
                saver.save(sess, savename)
            myepoch = epoch
        coordinator.join(threads)
    print(counter)
    return None


if __name__ == "__main__":
    Train('/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC/train',
        '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC/val',
        '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC/test')
