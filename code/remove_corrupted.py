import tensorflow as tf
import glob
import os
pth = ['/Users/ashwin/yt8m/v2/video/test/*.tfrecord', '/Users/ashwin/yt8m/v2/video/train/*.tfrecord', '/Users/ashwin/yt8m/v2/video/validate/*.tfrecord']

for path in pth:
    files =  glob.glob(path) 

    filesSize = len(files)
    print(filesSize)
    cnt = 0 

    for filename in files:
        cnt = cnt + 1
        print('checking %d/%d %s' % (cnt, filesSize, filename))
        try:
            for example in tf.python_io.tf_record_iterator(filename): 
                tf_example = tf.train.Example.FromString(example) 
        except :
            print("removing %s" % filename)
            os.remove(filename)