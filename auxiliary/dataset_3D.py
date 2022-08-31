from __future__ import print_function
import torch.utils.data as data
import os.path
import torchvision.transforms as transforms
from PIL import Image
from utils import *
# import meshio
import meshio_custom
import time
import glob
import math
import sklearn.preprocessing as sklp

class ShapeNet(data.Dataset):
    def __init__(self,
                 rootimg='./data/ShapeNet/ShapeNetRendering',
                 rootpc="./data/customShapeNet_mat",
                 class_choice = "lumbar_vertebra_05",
                 train = True, npoints = 2500, normal = False,
                 SVR=False, idx=0, extension = 'png'):
        self.normal = normal
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join('./data/synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        self.idx=idx
        self.extension = extension
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            dir_img  = os.path.join(self.rootimg, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))

            try:
                dir_point = os.path.join(self.rootpc, self.cat[item])
                fns_pc = sorted(os.listdir(dir_point))
            except:
                fns_pc = []
            fns = [val for val in fns_img if val + '.obj' in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_img)), "%"),
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    self.meta[item].append( ( os.path.join(dir_img, fn, "rendering"), os.path.join(dir_point, fn + '.obj'),
                                              item, fn ) )
            else:
                empty.append(item)
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        self.train_transforms = transforms.Compose([
                            transforms.Resize(size =  256, interpolation = 2),
                            # transforms.RandomCrop((224, 224)),
                            # transforms.ColorJitter(brightness=.4, contrast=.8, saturation=.4),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(size=256, interpolation=2),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.ToTensor(),
        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()

    def __getitem__(self, index):
        fn = self.datapath[index]

        # Obj format
        # mesh = meshio.read(fn[1])
        # points = mesh.points

        mesh = meshio_custom.read_obj(fn[1])
        points_origin = mesh['vertices']
        faces_origin = mesh['faces']

        points_max = np.max(points_origin, axis=0)
        points_min = np.min(points_origin, axis=0)
        points_origin = (points_origin - points_min) / (points_max - points_min)
        points_origin = 2. * points_origin - 1.

        # print(np.max(points_origin, axis=0))
        # print(np.min(points_origin, axis=0))

        indices = np.random.randint(points_origin.shape[0], size=self.npoints)
        points_sampled = points_origin[indices,:]
        faces_sampled = faces_origin[indices,:]
        if self.normal:
            # normals = mesh.get_cells_type("triangle")

            normals_sampled = np.zeros([self.npoints, 3])
            for i in range(0, self.npoints):
                v10 = points_origin[faces_sampled[i, 1]] - points_origin[faces_sampled[i, 0]]
                v20 = points_origin[faces_sampled[i, 2]] - points_origin[faces_sampled[i, 0]]
                normals_sampled[i, :] = np.cross(v10, v20)
            normals_sampled = sklp.normalize(normals_sampled, axis=1)

            '''
            normals_origin = np.zeros(points_origin.shape)
            v10 = points_origin[faces_origin[:, 1]] - points_origin[faces_origin[:, 0]]
            v20 = points_origin[faces_origin[:, 2]] - points_origin[faces_origin[:, 0]]
            normals_origin_value = np.cross(v10, v20)
            normals_origin[faces_origin[:,0]] += normals_origin_value[:]
            normals_origin[faces_origin[:,1]] += normals_origin_value[:]
            normals_origin[faces_origin[:,2]] += normals_origin_value[:]
            normals_origin_len = np.sqrt(normals_origin[:,0]*normals_origin[:,0]+normals_origin[:,1]*normals_origin[:,1]+normals_origin[:,2]*normals_origin[:,2])
            normals_origin = normals_origin / normals_origin_len.reshape(-1, 1)
            normals_sampled = normals_origin[indices,:]
            '''

        else:
            normals_sampled = 0

        # Resize to np.array([10000, 3])
        # points_origin = np.concatenate([points_origin, np.zeros([(15000 - points_origin.shape[0]), 3])], axis=0)
        # points_origin = np.zeros([1])
        # print(points_origin.shape)

        cat = fn[2]
        name = fn[3]
        if self.SVR:
            files = glob.glob(os.path.join(fn[0], '*.%s' % self.extension))
            files = sorted(files)
            num_files = len(files)
            stack_data = np.zeros([num_files, 256, 256])
            for idx in range(0, num_files):
                filename = files[idx]
                image = Image.open(filename)
                image = image.resize([256, 256])
                if self.train:
                    data = self.train_transforms(image)
                else:
                    data = self.val_transforms(image)
                data = data[0, :, :]
                stack_data[idx] = data

            vol_data = np.zeros([256, 256, 256])
            for v_idx in range(0, 256):
                m_idx = max(0, math.ceil(float(v_idx) * float(num_files) / 256.) - 1)
                vol_data[v_idx] = stack_data[m_idx]
        else:
            vol_data = 0
        return vol_data, points_sampled, normals_sampled, faces_sampled, points_origin, name, cat

    def __len__(self):
        return len(self.datapath)


if __name__  == '__main__':
    print('Testing Shapenet dataset')
    dataset  =  ShapeNet(class_choice = None,
                 train = True, npoints = 10000, normal = True,
                 SVR=True, idx=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=int(12))
    time1 = time.time()
    for i, data in enumerate(dataloader, 0):
        img, points, normals, name, cat = data
        print(img.shape)
        print(points.shape,normals.shape)
        print(cat[0],name[0],points.max(),points.min())
    time2 = time.time()
    print(time2-time1)
