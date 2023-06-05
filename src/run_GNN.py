import time
import os
import argparse
import numpy as np
import torch
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from GNN import GNN
from GNN_early import GNNEarly
from data import get_dataset, set_train_val_test_split
from heterophilic import get_fixed_splits
from graph_rewiring import apply_beltrami

from gread_params import best_params_dict, hetero_params, shared_gread_params, shared_grand_params
from utils import dirichlet_energy
import wandb


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat, pos_encoding)
    
    if model.opt['save_result'] == True:
        save_what = 'attention'
        # SA
        if save_what == 'attention' and model.opt['block'] == 'attention':
            from torch_geometric.utils import to_dense_adj
            att_ws = to_dense_adj(edge_index=model.odeblock.odefunc.edge_index, edge_attr=model.odeblock.odefunc.attention_weights.mean(dim=1))
            import numpy as np; np.save(r"output_adj_weights/{}_att_ws.npy".format(model.opt['dataset']),att_ws.cpu().detach().numpy())
            np.save(r"output_adj_weights/{}_att_ws_sp.npy".format(model.opt['dataset']),model.odeblock.odefunc.attention_weights.mean(dim=1).cpu().detach().numpy())
        elif save_what =='features':
            import numpy as np; np.save(r"output_features/{}.npy".format(model.opt['dataset']),out.cpu().detach().numpy())

    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

    if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        regularization_coeffs = model.regularization_coeffs

        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )
        loss = loss + reg_loss

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()

def train_baseline(model, optimizer, data, pos_encoding=None):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    feat = data.x
    if model.opt['use_labels']:
        train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

        feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
    else:
        train_pred_idx = data.train_mask

    out = model(feat, pos_encoding)

    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    epoch = model.odeblock.odefunc.epoch
    if opt['wandb']:
        # wandb tracking
        # need to calc loss again
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(logits[data.train_mask], data.y.squeeze()[data.train_mask])
        wandb_log(data, model, opt, loss, accs[0], accs[1], accs[2], epoch)
        model.odeblock.odefunc.wandb_step = 0  # resets the wandbstep counter in function after eval forward pass

    return accs

@torch.no_grad()
def test_baseline(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
    model.eval()
    feat = data.x
    if model.opt['use_labels']:
        feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
    logits, accs = model(feat, pos_encoding), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    if opt['wandb']:
        lf = torch.nn.CrossEntropyLoss()
        loss = lf(logits[data.train_mask], data.y.squeeze()[data.train_mask])
    return accs

@torch.no_grad()
def calc_energy_homoph(data, model, opt):
    # every epoch stats for greed linear and non linear
    num_nodes = data.num_nodes

    x0 = model.encoder(data.x)
    T0_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, x0)

    x0r = x0 / torch.norm(x0, p='fro')
    T0r_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, x0r)

    xN = model.forward_XN(data.x)
    TN_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, xN)

    xNr = xN / torch.norm(xN, p='fro')
    TNr_dirichlet = dirichlet_energy(data.edge_index, data.edge_attr, num_nodes, xNr)
    return T0_dirichlet, T0r_dirichlet, TN_dirichlet, TNr_dirichlet

def wandb_log(data, model, opt, loss, train_acc, val_acc, test_acc, epoch):
    model.eval()

    wandb.log({"loss": loss,
                "forward_nfe": model.fm.sum, "backward_nfe": model.bm.sum,
                "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                "epoch_step": epoch
                })

    return

def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)

def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['beltrami']:
        opt['beltrami'] = True

def main(cmd_opt):
    if cmd_opt['use_best_params']:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        # merge_cmd_args(cmd_opt, opt)
    else:
        opt = cmd_opt

    if opt['function'] == 'gread':
        opt = shared_gread_params(opt)
    elif opt['function'] == 'laplacian':
        opt = shared_grand_params(opt)
    opt = hetero_params(opt)


    if opt['wandb']:
        os.environ["WANDB_MODE"] = "run"
    else:
        os.environ["WANDB_MODE"] = "disabled"
    
    GPU_NUM = opt['gpu']
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device

    if opt['wandb']:
        if 'wandb_run_name' in opt.keys():
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], config=opt, allow_val_change=True, name=opt['wandb_run_name'])
        else:
            wandb_run = wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], config=opt, allow_val_change=True)
        opt = wandb.config
        wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log

    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    if opt['hetero_SL']:
        dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
    if opt['hetero_undir']:
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    this_test = test
    results = []
    for rep in range(opt['num_splits']):
        print(f"rep {rep}")
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
            dataset.data = set_train_val_test_split(np.random.randint(0, 1000), dataset.data,
                                                    num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500,
                                                    num_per_class=opt['num_train_per_class'])
        if opt['geom_gcn_splits']:
            if opt['dataset'] == "Citeseer":
                dataset = get_dataset(opt, '../data', opt['not_lcc']) #geom-gcn citeseer uses splits over LCC and not_LCC so need to reload each rep/split
            data = get_fixed_splits(dataset.data, opt['dataset'], rep)
            dataset.data = data

        if opt['beltrami']:
            pos_encoding = apply_beltrami(dataset.data, opt).to(device)
            if opt['wandb']:
                wandb.config.update({'pos_enc_dim': pos_encoding.shape[1]}, allow_val_change=True)
            else:
                opt['pos_enc_dim'] = pos_encoding.shape[1]
        else:
            pos_encoding = None

        data = dataset.data.to(device)
        if opt['function'] in ['laplacian', 'gread']:
            model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)
        else:
            raise Exception('Unknown function')

        parameters = [p for p in model.parameters() if p.requires_grad]
        print(opt)
        print_model_params(model)
        optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
        best_time = best_epoch = train_acc = val_acc = test_acc = 0
        if opt['patience'] is not None:
            patience_count = 0
        for epoch in range(1, opt['epoch']):
            start_time = time.time()
            loss = train(model, optimizer, data, pos_encoding)

            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

            best_time = opt['time']
            if tmp_val_acc > val_acc:
                best_epoch = epoch
                train_acc = tmp_train_acc
                val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                best_time = opt['time']
                patience_count = 0
            else:
                patience_count += 1
            if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
                best_epoch = epoch
                val_acc = model.odeblock.test_integrator.solver.best_val
                test_acc = model.odeblock.test_integrator.solver.best_test
                train_acc = model.odeblock.test_integrator.solver.best_train
                best_time = model.odeblock.test_integrator.solver.best_time

            print(f"Epoch: {epoch}, Runtime: {time.time() - start_time:.4f}, Loss: {loss:.3f}, "
                f"forward nfe {model.fm.sum}, backward nfe {model.bm.sum}, "
                f"tmp_train: {tmp_train_acc:.4f}, tmp_val: {tmp_val_acc:.4f}, tmp_test: {tmp_test_acc:.4f}, "
                f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Best time: {best_time:.4f}")

            if np.isnan(loss):
                wandb_run.finish()
                break
            if opt['patience'] is not None:
                if patience_count >= opt['patience']:
                    break
        print(
            f"best val accuracy {val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch} and best time {best_time:2f}")

        if opt['num_splits'] > 1:
            results.append([test_acc, val_acc, train_acc])

    if opt['num_splits'] > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
        results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                             'test_acc_std': test_acc_std}
    else:
        results = {'test_mean': test_acc, 'val_mean': val_acc, 'train_mean': train_acc}
    print(results)
    if opt['wandb']:
        wandb.log(results)
        wandb_run.finish()
    return train_acc, val_acc, test_acc
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #run args
    parser.add_argument('--use_best_params', action='store_true', help="flag to take the best params of GREAD")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--patience', type=int, default=None, help='set if training should use patience on val acc')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')

    # data args
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw', help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=0.0,help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5, help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true', help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true', help='use the 10 fixed splits from https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1, help='the number of splits to repeat the results on')
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--hetero_SL', action='store_true', help='control self loops for Chameleon/Squirrel')
    parser.add_argument('--hetero_undir', action='store_true', help='control undirected for Chameleon/Squirrel')

    # GNN args
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='gread', help='laplacian, transformer, greed, GAT')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
    parser.add_argument('--fc_out', type=eval, default=False, help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--source_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) source')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--use_mlp', type=eval, default=False, help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', type=eval, default=False, help='If try get rid of alpha param and the source*x0 source term')
    parser.add_argument('--XN_activation', type=eval, default=False, help='whether to relu activate the terminal state')
    parser.add_argument('--m2_mlp', type=eval, default=False, help='whether to use decoder mlp')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', type=eval, default=False, help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint, symplectic_euler, leapfrog")
    parser.add_argument('--step_size', type=float, default=0.1, help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun", help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true', help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1, help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0, help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000, help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument('--no_early', type=eval, default=True)
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100, help="Maximum number steps for the dopri5Early test integrator. used if getting OOM errors at test time")

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")
    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')
    
    # GCN ablation args
    parser.add_argument('--gcn_fixed', type=str, default='False', help='fixes layers in gcn')
    parser.add_argument('--gcn_enc_dec', type=str, default='False', help='uses encoder decoder with GCN')
    parser.add_argument('--gcn_non_lin', type=str, default='False', help='uses non linearity with GCN')
    parser.add_argument('--gcn_symm', type=str, default='False', help='make weight matrix in GCN symmetric')
    parser.add_argument('--gcn_bias', type=str, default='False', help='make GCN include bias')
    parser.add_argument('--gcn_mid_dropout', type=str, default='False', help='dropout between GCN layers')
    parser.add_argument('--gcn_params', nargs='+', default=None, help='list of args for gcn ablation')
    parser.add_argument('--gcn_params_idx', type=int, default=0, help='index to track GCN ablation')

    # rewiring args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")
    
    # beltrami args
    parser.add_argument('--beltrami', type=eval, default=False, help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', type=eval, default=False, help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="GDC",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                        help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                        help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    
    # gread args
    parser.add_argument('--reaction_term', type=str, default='bspm', help='bspm, fisher, allen-cahn')
    parser.add_argument('--beta_diag', type=eval, default=False)
    
    # with source term args
    parser.add_argument('--trusted_mask', type=eval, default=False, help='mask')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--noise_pos', type=str, help='all, test')
    parser.add_argument('--prediffuse', action='store_true')
    parser.add_argument('--x0', action='store_true')
    parser.add_argument('--nox0', type=eval, default=False)
    parser.add_argument('--icxb', type=float, default=1.0)
    parser.add_argument('--source_scale', type=float, default=1.0)
    parser.add_argument('--alltime', action='store_true')
    parser.add_argument('--allnumtrain', action='store_true')
    parser.add_argument("--num_train_per_class", type=int, default=20)

    # visualize-related args
    parser.add_argument('--save_result', type=eval, default=False, help='')

    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_sweep', action='store_true',help="flag if sweeping")
    parser.add_argument('--wandb_entity', default="username", type=str)
    parser.add_argument('--wandb_project', default="example", type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
    parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")
    args = parser.parse_args()
    opt = vars(args)
    main(opt)