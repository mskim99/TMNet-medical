from __future__ import print_function
import torch.utils.data as data
import os.path
import torchvision.transforms as transforms
from PIL import Image
from utils import *
import meshio
import time
import glob
import math

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

        self.transforms = transforms.Compose([
                             transforms.Resize(size =  224, interpolation = 2),
                             transforms.ToTensor(),
                        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()

    def __getitem__(self, index):
        fn = self.datapath[index]

        # Obj format
        mesh = meshio.read(fn[1])
        points = mesh.points
        points_max = np.max(points)
        points = points / points_max
        indices = np.random.randint(points.shape[0], size=self.npoints)
        points = points[indices,:]
        if self.normal:
            normals = mesh.get_cells_type("triangle")
            normals = normals[indices,:]
        else:
            normals = 0

        cat = fn[2]
        name = fn[3]
        if self.SVR:
            files = glob.glob(os.path.join(fn[0], '*.%s' % self.extension))
            files = sorted(files)
            num_files = len(files)
            stack_data = np.zeros([num_files, 224, 224])
            for idx in range(0, num_files):
                filename = files[idx]
                image = Image.open(filename)
                image = image.resize([137, 137])
                data = self.transforms(image)
                data = data[0, :, :]
                stack_data[idx] = data

            vol_data = np.zeros([224, 224, 224])
            for v_idx in range(0, 224):
                m_idx = max(0, math.ceil(float(v_idx) * float(num_files) / 224.) - 1)
                print(str(v_idx) + ' ' + str(m_idx))
                vol_data[v_idx] = stack_data[m_idx]
        else:
            data = 0
        return data, points, normals, name, cat

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
