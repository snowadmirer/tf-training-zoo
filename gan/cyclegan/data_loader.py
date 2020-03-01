import scipy
import scipy.misc
from glob import glob
import numpy as np
import os

class DataLoader():
    def __init__(self, dataset_dirs, img_res=(128, 128)):
        self.dataset_dirs = dataset_dirs
        self.img_res = img_res
        self.path = {}
        self.path['trainA'] = []
        self.path['trainB'] = []
        self.path['testA'] = []
        self.path['testB'] = []
        for dataset_dir in self.dataset_dirs:
            self.path['trainA'].extend(glob('%s/*' % os.path.join(dataset_dir, 'trainA')))
            self.path['trainB'].extend(glob('%s/*' % os.path.join(dataset_dir, 'trainB')))
            self.path['testA'].extend(glob('%s/*' % os.path.join(dataset_dir, 'testA')))
            self.path['testA'].extend(glob('%s/*' % os.path.join(dataset_dir, 'testA')))
        self.numA = len(self.path['trainA'])
        self.numB = len(self.path['trainB'])
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = self.path[data_type]

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = self.path[data_type + 'A']
        path_B = self.path[data_type + 'B']

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    dataset_dirs = ['man2woman',]
    data = DataLoader(dataset_dirs)
    imgs_A, imgs_B = next(data.load_batch())
    print(imgs_A.shape)
    print(imgs_B.shape)
