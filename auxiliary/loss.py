import torch
import numpy as np

import utils

def normalize(input, p=2, dim=1, eps=1e-12):
    input = input.float()
    return input / input.norm(p, dim).clamp(min=eps).unsqueeze(dim).expand_as(input)


def calculate_l2_loss(x, y):
    assert x.size() == y.size()
    ret = torch.mean(torch.pow((x - y), 2))
    return ret


def get_edge_loss(vertices,faces):
    # bs*2562*3 bs*5120*3
    edges = torch.cat((faces[:,:,:2],faces[:,:,[0,2]],faces[:,:,1:]),1) # bs * (3*5120) *2
    edges_vertices = vertices.index_select(1,edges.view(-1)).\
        view(vertices.size(0)*edges.size(0),edges.size(1),edges.size(2),vertices.size(2)) #
    indices = (torch.arange(0,vertices.size(0))*(1+vertices.size(0))).type(torch.cuda.LongTensor)
    edges_vertices = edges_vertices.index_select(0,indices) # bs * (3*5120) *2 *3
    edges_len = torch.norm((edges_vertices[:,:,0]-edges_vertices[:,:,1]),2,2)
    edges_len = torch.pow(edges_len,2)
    nonzero = len(edges_len.nonzero())
    edge_loss = torch.sum(edges_len)/nonzero
    return edge_loss


def get_edge_loss_stage1(vertices,edge):
    # vertices bs*points_number*3 edge edge_number*2
    vertices_edge = vertices.index_select(1,edge.view(-1)).\
        view(vertices.size(0),edge.size(0),edge.size(1),vertices.size(2))
    vertices_edge_vector = vertices_edge[:,:,0] - vertices_edge[:,:,1]
    vertices_edge_len = torch.pow(vertices_edge_vector.norm(2,2),2)
    edge_loss = torch.mean(vertices_edge_len)
    return edge_loss


def get_edge_loss_stage1_whmr(vertices, vertices_gt, edge, edge_gt):

    el = vertices[:, edge[:, 1], :] - vertices[:, edge[:, 0], :]
    edges_length = torch.sqrt(el[:, :, 0] * el[:, :, 0] + el[:, :, 1] * el[:, :, 1] + el[:, :, 2] * el[:, :, 2])

    elg = vertices_gt[:, edge_gt[:, 1], :] - vertices_gt[:, edge_gt[:, 0], :]
    edge_length_gt_mean = torch.sqrt(elg[:, :, 0] * elg[:, :, 0] + elg[:, :, 1] * elg[:, :, 1] + elg[:, :, 2] * elg[:, :, 2])
    edge_length_gt_mean = torch.sum(edge_length_gt_mean) / float(edge_gt.size(0))

    edge_loss = torch.abs(edges_length - edge_length_gt_mean)
    edge_loss = torch.sum(edge_loss)

    return edge_loss


def get_normal_loss(vertices, faces, gt_normals, idx2):
    idx2 = idx2.type(torch.cuda.LongTensor).detach()
    edges = torch.cat((faces[:,:,:2],faces[:,:,[0,2]],faces[:,:,1:]),1)
    edges_vertices = vertices.index_select(1,edges.view(-1)).\
        view(vertices.size(0)*edges.size(0),edges.size(1),edges.size(2),vertices.size(2))
    indices = (torch.arange(0,vertices.size(0))*(1+vertices.size(0))).type(torch.cuda.LongTensor)
    edges_vertices = edges_vertices.index_select(0,indices)
    edges_len1 = edges_vertices[:,:,0] - edges_vertices[:,:,1]
    edges_len2 = edges_vertices[:,:,1] - edges_vertices[:,:,0]
    edges_vector = torch.stack((edges_len1,edges_len2),2)
    gt_normals = gt_normals.index_select(1, idx2.contiguous().view(-1)).contiguous().view(gt_normals.size(0) * idx2.size(0),
                                                                                          idx2.size(1), gt_normals.size(2))
    gt_normals = gt_normals.index_select(0, indices)
    gt_normals_edges = gt_normals.index_select(1, edges.view(-1)).view(gt_normals.size(0) * edges.size(0),
                                                                       edges.size(1), edges.size(2), gt_normals.size(2))
    gt_normals_edges = gt_normals_edges.index_select(0, indices)
    gt_normals_edges = normalize(gt_normals_edges, p=2, dim=3)
    edges_vector = normalize(edges_vector,p=2,dim=3)
    cosine = torch.abs(torch.sum(torch.mul(edges_vector, gt_normals_edges), 3))
    nonzero = len(cosine.nonzero())
    normal_loss = torch.sum(cosine)/nonzero

    return normal_loss


def get_normal_loss_mdf(gen_normals, gt_normals, idx2):

    normal_loss = 0.0
    for i in range(0, gen_normals.shape[0]):
        diff = gen_normals[i, :] - gt_normals[0, idx2[0, i], :]
        diff = torch.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        normal_loss = normal_loss + diff

    normal_loss = normal_loss / gen_normals.shape[0]
    return normal_loss


def smoothness_loss_parameters(faces):
    # faces faces_number*3(array)
    print('calculating the smoothness loss parameters, gonna take a few moments')
    if hasattr(faces, 'get'):
        faces = faces.get()
    vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))])) # edges

    v0s = np.array([v[0] for v in vertices], 'int32')
    v1s = np.array([v[1] for v in vertices], 'int32')
    v2s = []
    v3s = []
    for v0, v1 in zip(v0s, v1s):
        count = 0
        for face in faces:
            if v0 in face and v1 in face:
                v = np.copy(face)
                v = v[v != v0]
                v = v[v != v1]
                if count == 0:
                    v2s.append(int(v[0]))
                    count += 1
                else:
                    v3s.append(int(v[0]))
        if len(v3s) < len(v2s):
            v3s.append(0)

    v2s = np.array(v2s, 'int32')
    v3s = np.array(v3s, 'int32')
    print('calculated')
    return v0s, v1s, v2s, v3s


def get_smoothness_loss_stage1(vertices, parameters, eps=1e-6):
    # make v0s, v1s, v2s, v3s
    # vertices (bs*num_points*3)
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.size(0)

    v0s = torch.from_numpy(v0s).type(torch.cuda.LongTensor)
    v1s = torch.from_numpy(v1s).type(torch.cuda.LongTensor)
    v2s = torch.from_numpy(v2s).type(torch.cuda.LongTensor)
    v3s = torch.from_numpy(v3s).type(torch.cuda.LongTensor)

    v0s = vertices.index_select(1, v0s)
    v1s = vertices.index_select(1, v1s)
    v2s = vertices.index_select(1, v2s)
    v3s = vertices.index_select(1, v3s)

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = torch.sum(a1.pow(2),dim=2)
    b1l2 = torch.sum(b1.pow(2),dim=2)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1*b1,dim=2)

    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1.pow(2) + eps)
    c1 = a1 * (((ab1/(a1l2+eps)).unsqueeze(2)).expand_as(a1))

    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2.pow(2),dim=2)
    b2l2 = torch.sum(b2.pow(2),dim=2)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2*b2,dim=2)

    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2.pow(2) + eps)
    c2 = a2 * (((ab2 / (a2l2 + eps)).unsqueeze(2)).expand_as(a2))

    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = torch.sum(cb1*cb2, dim=2) / (cb1l1 * cb2l1 + eps)
    loss = torch.sum((cos+1).pow(2)) / batch_size

    return loss


def get_smoothness_loss(vertices, parameters,faces_bn,eps=1e-6):
    # make v0s, v1s, v2s, v3s
    # vertices (bs*num_points*3)
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.size(0)

    v0s = torch.from_numpy(v0s).type(torch.cuda.LongTensor)
    v1s = torch.from_numpy(v1s).type(torch.cuda.LongTensor)
    v2s = torch.from_numpy(v2s).type(torch.cuda.LongTensor)
    v3s = torch.from_numpy(v3s).type(torch.cuda.LongTensor)

    vs = torch.stack((v0s,v1s,v2s,v3s),1)

    faces_bn_view = faces_bn.view(faces_bn.size(0),-1)
    faces_bn_view = faces_bn_view.sort(1)[0]

    count = torch.ones(1).expand_as(faces_bn_view).type(torch.cuda.LongTensor)
    count_sum = torch.zeros(1).expand((faces_bn.size(0),vertices.shape[1])).type(torch.cuda.LongTensor)

    count_sum = count_sum.scatter_add(1, faces_bn_view, count)
    count_sum = (count_sum > 0).type(torch.cuda.ByteTensor)

    b1 = count_sum.ne(1).type(torch.cuda.LongTensor)

    i2 = vs.expand([faces_bn.size(0),vs.size(0),vs.size(1)])
    i2_unrolled = i2.view(i2.size()[0],-1)
    out_mask = torch.gather(b1,1,i2_unrolled).resize_as_(i2)
    zero_mask = out_mask.sum(2,keepdim=True).long().eq(0).long().expand_as(out_mask)
    final = i2 * zero_mask

    v0s_bn = final[:,:,0]
    v1s_bn = final[:,:,1]
    v2s_bn = final[:,:,2]
    v3s_bn = final[:,:,3]

    v0s = vertices.index_select(1, v0s_bn.view(-1)).view(vertices.size(0)*v0s_bn.size(0),v0s_bn.size(1),vertices.size(2))
    v1s = vertices.index_select(1, v1s_bn.view(-1)).view(vertices.size(0)*v1s_bn.size(0),v1s_bn.size(1),vertices.size(2))
    v2s = vertices.index_select(1, v2s_bn.view(-1)).view(vertices.size(0)*v2s_bn.size(0),v2s_bn.size(1),vertices.size(2))
    v3s = vertices.index_select(1, v3s_bn.view(-1)).view(vertices.size(0)*v3s_bn.size(0),v3s_bn.size(1),vertices.size(2))

    indices = (torch.arange(0, vertices.size(0)) * (1 + vertices.size(0))).type(torch.cuda.LongTensor)

    v0s = v0s.index_select(0,indices)
    v1s = v1s.index_select(0,indices)
    v2s = v2s.index_select(0,indices)
    v3s = v3s.index_select(0,indices)

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = torch.sum(a1.pow(2),dim=2)
    b1l2 = torch.sum(b1.pow(2),dim=2)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1*b1,dim=2)

    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1.pow(2) + eps)
    c1 = a1 * (((ab1/(a1l2+eps)).unsqueeze(2)).expand_as(a1))
    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2.pow(2),dim=2)
    b2l2 = torch.sum(b2.pow(2),dim=2)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2*b2,dim=2)
    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2.pow(2) + eps)
    c2 = a2 * (((ab2 / (a2l2 + eps)).unsqueeze(2)).expand_as(a2))

    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = torch.sum(cb1*cb2, dim=2) / (cb1l1 * cb2l1 + eps)
    loss = torch.sum((cos+1).pow(2)) / batch_size
    return loss


def get_uniform_loss_global(vertices_gen, vertices_gt):

    # Division of 3D space
    # Using for uniform loss & detailed shape reconstruction
    b_range = np.array([[0.0, 0.5, 1.0 + 1e-5], [0.0, 0.5, 1.0 + 1e-5], [0.0, 0.5, 1.0 + 1e-5]])
    b_v_list_gt = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),),
                           dtype=object)
    b_f_list_gt = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),),
                           dtype=object)
    b_v_list_gen = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),),
                            dtype=object)
    b_f_list_gen = np.empty((((b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)),),
                            dtype=object)
    loss_part = np.zeros([(b_range[0].shape[0] - 1) * (b_range[1].shape[0] - 1) * (b_range[2].shape[0] - 1)])

    orig_v_gen_num = float(vertices_gen.shape[0])
    orig_v_gt_num = float(vertices_gt.shape[0])

    '''
    print(orig_v_gen_num)
    print(orig_v_gt_num)
    '''

    b_v_idx = 0
    for x_i in range(0, (b_range[0].shape[0] - 1)):
        for y_i in range(0, (b_range[1].shape[0] - 1)):
            for z_i in range(0, (b_range[2].shape[0] - 1)):

                # Comparison between vertices of ground truth mesh
                b_v_list_gt[b_v_idx] = []
                b_f_list_gt[b_v_idx] = []
                for e_i in range(0, vertices_gt.shape[0]):
                    if b_range[0][x_i] <= vertices_gt[e_i][0] < b_range[0][x_i + 1] and \
                            b_range[1][y_i] <= vertices_gt[e_i][1] < b_range[0][y_i + 1] and \
                            b_range[2][z_i] <= vertices_gt[e_i][2] < b_range[2][z_i + 1]:
                        b_v_list_gt[b_v_idx].append(vertices_gt[e_i][:])
                        b_f_list_gt[b_v_idx].append(e_i)
                b_v_list_gt[b_v_idx] = np.array(b_v_list_gt[b_v_idx])

                # Comparison between vertices of generated mesh
                b_v_list_gen[b_v_idx] = []
                b_f_list_gen[b_v_idx] = []
                for e_i in range(0, vertices_gen.shape[0]):
                    if b_range[0][x_i] <= vertices_gen[e_i][0] < b_range[0][x_i + 1] and \
                            b_range[1][y_i] <= vertices_gen[e_i][1] < b_range[0][y_i + 1] and \
                            b_range[2][z_i] <= vertices_gen[e_i][2] < b_range[2][z_i + 1]:
                        b_v_list_gen[b_v_idx].append(vertices_gen[e_i][:])
                        b_f_list_gen[b_v_idx].append(e_i)
                b_v_list_gen[b_v_idx] = np.array(b_v_list_gen[b_v_idx])

                b_v_idx = b_v_idx + 1

    v_num_gt_total = 0
    v_num_gen_total = 0
    b_v_num_valid = b_v_idx
    for i in range(0, b_v_idx):

        v_num_gt = float(b_v_list_gt[i].shape[0])
        v_num_gen = float(b_v_list_gen[i].shape[0])
        v_num_gt_total = v_num_gt_total + v_num_gt
        v_num_gen_total = v_num_gen_total + v_num_gen

        '''
        print('Before')
        print(v_num_gt)
        print(v_num_gen)
        '''

        if orig_v_gt_num > orig_v_gen_num:
            v_num_gt = v_num_gt * (orig_v_gen_num / orig_v_gt_num)
        elif orig_v_gt_num < orig_v_gen_num:
            v_num_gen = v_num_gen * (orig_v_gt_num / orig_v_gen_num)

        if v_num_gt > 0.:
            loss_part[i] = (v_num_gen - v_num_gt) * (v_num_gen - v_num_gt) / v_num_gt
        else:
            loss_part[i] = 0.
            b_v_num_valid = b_v_num_valid - 1

        '''
        print('After')
        print(v_num_gt)
        print(v_num_gen)
        print(loss_part[i])
        '''

    gu_loss = np.sum(loss_part) / float(b_v_num_valid)

    '''
    print('Total')
    print(v_num_gt_total)
    print(v_num_gen_total)
    print(gu_loss)
    '''

    return gu_loss, b_v_list_gen, b_f_list_gen


def get_uniform_loss_local(b_v_list_gen, b_f_list_gen):

    num_vertices = b_v_list_gen.shape[0]
    b_v_list_gen_edges = utils.get_edges(b_f_list_gen)
    b_e_gen = b_v_list_gen[b_v_list_gen_edges[:, 1], :] - b_v_list_gen_edges[b_v_list_gen_edges[:, 0], :]
    edges_length = torch.sqrt(b_e_gen[:, 0] * b_e_gen[:, 0] + b_e_gen[:, 1] * b_e_gen[:, 1] + b_e_gen[:, 2] * b_e_gen[:, 2])

    exp_dist = np.sqrt(2 * 3.141592 * 0.25 / float(num_vertices) * 1.73205) # div 222, root 3

    lu_loss = (edges_length - exp_dist) * (edges_length - exp_dist) / exp_dist
    lu_loss = torch.sum(lu_loss)

    return lu_loss