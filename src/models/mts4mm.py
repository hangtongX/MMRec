import copy
import hashlib
import os
import tempfile

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender


class MTS4MM(GeneralRecommender):
    TEACHER_NAMES = ('ui', 'img', 'txt')

    STUDENT_FUSION_COEFF = 0.4

    FIXED_TEACHER_SETTINGS = {
        'use_kd': True,
        'ema_mu': 0.995,
        'kd_warmup_epochs': 10,
        'kd_ramp_epochs': 10,
        'kd_emb_weight': 0.0,
        'kd_emb_on_users': False,
    }

    def __init__(self, config, dataset):
        super(MTS4MM, self).__init__(config, dataset)
        self.epoch = 0
        self.n_ui_layers = int(config['n_ui_layers'])
        self.embedding_dim = int(config['embedding_size'])
        self.knn_k = int(config['knn_k'])
        self.n_layers = int(config['n_layers'])
        self.reg_weight = float(config['reg_weight'])
        self.cl_weight = float(config['cl_loss'])
        teacher_settings = self.FIXED_TEACHER_SETTINGS
        self.use_kd = teacher_settings['use_kd']
        self.ema_mu = teacher_settings['ema_mu']
        self.kd_warmup_epochs = teacher_settings['kd_warmup_epochs']
        self.kd_ramp_epochs = teacher_settings['kd_ramp_epochs']
        self.kd_rank_weight = float(config['kd_rank_weight'])
        self.kd_tau = float(config['kd_tau'])
        self.freeze_feat_emb = bool(config['freeze_feat_emb'])
        if not 0 <= self.ema_mu < 1:
            raise ValueError('ema_mu must be in [0, 1); use 0 for the latest online experts')
        if self.kd_tau <= 0:
            raise ValueError('Temperatures must be positive')
        if self.n_ui_layers < 1:
            raise ValueError('n_ui_layers must be positive for UI layer averaging over 1..L')
        if self.knn_k < 1:
            raise ValueError('knn_k must be positive')
        if self.n_layers != 1:
            raise ValueError('MSCA student and experts require n_layers=1')

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        norm_adj = self.get_adj_mat()
        incidence = self.R
        del self.R
        self.register_buffer('norm_adj', self.sparse_mx_to_torch_sparse_tensor(norm_adj), persistent=False)
        self.register_buffer('R', self.sparse_mx_to_torch_sparse_tensor(incidence), persistent=False)

        dataset_path = os.path.abspath(os.path.join(config['data_path'], config['dataset']))
        for name, features in (('image', self.v_feat), ('text', self.t_feat)):
            embedding, projection, graph = None, None, None
            if features is not None:
                if features.shape[0] != self.n_items:
                    raise ValueError(name + ' feature rows do not match item count')

                features = features.float()
                embedding = nn.Embedding.from_pretrained(features, freeze=self.freeze_feat_emb)
                projection = nn.Linear(features.shape[1], self.embedding_dim)
                graph = self._load_modality_graph(F.normalize(features, p=2, dim=-1), dataset_path, name)
            setattr(self, name + '_embedding', embedding)
            setattr(self, name + '_trs', projection)
            self.register_buffer(name + '_original_adj', graph, persistent=False)

        self.v_feat = None if self.image_embedding is None else self.image_embedding.weight.detach()
        self.t_feat = None if self.text_embedding is None else self.text_embedding.weight.detach()
        self.softmax = nn.Softmax(dim=-1)
        self.query_common = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                          nn.Linear(self.embedding_dim, 1, bias=False))
        self.gate_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        self.t_user_embedding = copy.deepcopy(self.user_embedding) if self.use_kd else None
        self.t_item_id_embedding = copy.deepcopy(self.item_id_embedding) if self.use_kd else None

        for name, gate in (('image', self.gate_v), ('text', self.gate_t)):
            source = getattr(self, name + '_embedding')
            available = self.use_kd and source is not None
            setattr(self, 't_' + name + '_item_embedding', copy.deepcopy(self.item_id_embedding) if available else None)
            setattr(self, 't_' + name + '_trs', copy.deepcopy(getattr(self, name + '_trs')) if available else None)
            setattr(self, 't_gate_' + ('v' if name == 'image' else 't'), copy.deepcopy(gate) if available else None)
            setattr(self, 't_' + name + '_embedding', copy.deepcopy(source) if available and self.use_kd and not self.freeze_feat_emb else None)

        self.ema_teachers = nn.ModuleDict()
        if self.use_kd:
            for name, modules in self._online_expert_modules().items():
                shadow = nn.ModuleDict({key: copy.deepcopy(module) for key, module in modules.items()})
                shadow.requires_grad_(False)
                shadow.eval()
                self.ema_teachers[name] = shadow
        self.register_buffer('ema_num_updates', torch.zeros((), dtype=torch.long))
        self._eval_cache = None

    @torch.no_grad()
    def _load_modality_graph(self, features, dataset_path, name):
        fingerprint = hashlib.sha256(str(tuple(features.shape)).encode('ascii'))
        for start in range(0, self.n_items, 1024):
            fingerprint.update(features[start:start + 1024].detach().cpu().contiguous().numpy().tobytes())
        k = min(self.knn_k, self.n_items)
        cache_name = 'mts4mm_{}_cosine_weighted_v1_k{}_{}.pt'.format(name, k, fingerprint.hexdigest())
        cache_path = os.path.join(dataset_path, cache_name)
        if os.path.isfile(cache_path):
            graph = torch.load(cache_path, map_location='cpu')
            if graph.is_sparse:
                graph = graph.coalesce()
                if graph.shape == (self.n_items, self.n_items) and torch.isfinite(graph.values()).all():
                    return graph.to(features.device)
        neighbours, edge_weights = [], []
        for start in range(0, self.n_items, 1024):
            similarities = features[start:start + 1024] @ features.t()
            values, indices = similarities.topk(k, dim=-1, sorted=False)
            neighbours.append(indices)
            edge_weights.append(values)
        columns = torch.cat(neighbours).reshape(-1)
        rows = torch.arange(self.n_items, device=features.device).repeat_interleave(k)
        weights = torch.cat(edge_weights).reshape(-1)
        degrees = features.new_zeros(self.n_items)
        degrees.scatter_add_(0, rows, weights)
        if (degrees < 0).any():
            raise ValueError('Cosine-weighted KNN requires nonnegative weighted degrees')
        inv_sqrt = torch.zeros_like(degrees)
        positive = degrees > 0
        inv_sqrt[positive] = degrees[positive].pow(-0.5)
        graph = torch.sparse_coo_tensor(torch.stack((rows, columns)), weights * inv_sqrt[rows] * inv_sqrt[columns],
                                        (self.n_items, self.n_items)).coalesce()
        handle, temporary = tempfile.mkstemp(prefix=cache_name, suffix='.tmp', dir=dataset_path)
        os.close(handle)
        try:
            torch.save(graph.cpu(), temporary)
            os.replace(temporary, cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return graph

    def get_adj_mat(self):
        interactions = self.interaction_matrix.tocsr()
        interactions.sum_duplicates()
        interactions.data.fill(1.0)
        adjacency = sp.bmat([[None, interactions], [interactions.T, None]], format='csr')
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degrees)
        np.divide(1.0, np.sqrt(degrees), out=inv_sqrt, where=degrees > 0)
        normalized = sp.diags(inv_sqrt).dot(adjacency).dot(sp.diags(inv_sqrt)).tocoo()
        self.R = normalized.tocsr()[:self.n_users, self.n_users:].tocoo()
        return normalized

    def sparse_mx_to_torch_sparse_tensor(self, matrix):
        matrix = matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
        return torch.sparse_coo_tensor(indices, torch.from_numpy(matrix.data), matrix.shape).coalesce().to(self.device)

    def _online_expert_modules(self):
        if not self.use_kd:
            return {}
        experts = {'ui': {'user_embedding': self.t_user_embedding,
                          'item_embedding': self.t_item_id_embedding}}
        for name, view in (('image', 'img'), ('text', 'txt')):
            if getattr(self, name + '_embedding') is None:
                continue
            modules = {
                'item_embedding': getattr(self, 't_' + name + '_item_embedding'),
                'projection': getattr(self, 't_' + name + '_trs'),
                'gate': getattr(self, 't_gate_' + ('v' if name == 'image' else 't'))}
            raw = getattr(self, 't_' + name + '_embedding')
            if raw is not None:
                modules['feature_embedding'] = raw
            experts[view] = modules
        return experts

    def _teacher_module_pairs(self):
        return [(self.ema_teachers[name][key], module)
                for name, modules in self._online_expert_modules().items()
                if name in self.ema_teachers for key, module in modules.items()]

    @torch.no_grad()
    def ema_update_teacher(self):
        if not self.use_kd:
            return
        for teacher, expert in self._teacher_module_pairs():
            for target, source in zip(teacher.parameters(), expert.parameters()):
                target.mul_(self.ema_mu).add_(source, alpha=1.0 - self.ema_mu)
        self.ema_num_updates.add_(1)

    def train(self, mode=True):
        result = super(MTS4MM, self).train(mode)
        self.ema_teachers.eval()
        self.clear_eval_cache()
        return result

    def _encode_ui(self, user_embedding, item_embedding, adj):
        embeddings = torch.cat((user_embedding.weight, item_embedding.weight))

        accumulated = torch.zeros_like(embeddings)
        for _ in range(self.n_ui_layers):
            embeddings = torch.sparse.mm(adj, embeddings)
            accumulated = accumulated + embeddings
        return accumulated / self.n_ui_layers

    def _encode_modality(self, raw_embedding, projection, gate, item_embedding, adjacency):
        if raw_embedding is None:
            return None
        initial_items = item_embedding.weight * gate(projection(raw_embedding.weight))
        items = torch.sparse.mm(adjacency, initial_items) if adjacency.is_sparse else adjacency @ initial_items
        return torch.cat((torch.sparse.mm(self.R, items), items))

    def encode_ui_student(self, adj=None):
        return self._encode_ui(self.user_embedding, self.item_id_embedding, self.norm_adj if adj is None else adj)

    def encode_image_student(self):
        return self._encode_modality(self.image_embedding, self.image_trs, self.gate_v, self.item_id_embedding, self.image_original_adj)

    def encode_text_student(self):
        return self._encode_modality(self.text_embedding, self.text_trs, self.gate_t, self.item_id_embedding, self.text_original_adj)

    def encode_ui_expert(self, adj=None):
        if self.t_user_embedding is None:
            return None
        return self._encode_ui(self.t_user_embedding, self.t_item_id_embedding, self.norm_adj if adj is None else adj)

    def _encode_content_expert(self, name):
        raw = getattr(self, name + '_embedding')
        if not self.use_kd or raw is None:
            return None
        expert_raw = getattr(self, 't_' + name + '_embedding')
        return self._encode_modality(
            raw if expert_raw is None else expert_raw, getattr(self, 't_' + name + '_trs'),
            getattr(self, 't_gate_' + ('v' if name == 'image' else 't')),
            getattr(self, 't_' + name + '_item_embedding'), getattr(self, name + '_original_adj'))

    def encode_image_expert(self):
        return self._encode_content_expert('image')

    def encode_text_expert(self):
        return self._encode_content_expert('text')

    @torch.no_grad()
    def encode_ui_teacher(self, adj=None):
        if 'ui' not in self.ema_teachers:
            return None
        modules = self.ema_teachers['ui']
        return self._encode_ui(modules['user_embedding'], modules['item_embedding'],
                               self.norm_adj if adj is None else adj)

    @torch.no_grad()
    def _encode_content_teacher(self, name):
        view = 'img' if name == 'image' else 'txt'
        if view not in self.ema_teachers:
            return None
        modules = self.ema_teachers[view]

        raw = modules['feature_embedding'] if 'feature_embedding' in modules else getattr(self, name + '_embedding')
        return self._encode_modality(raw, modules['projection'], modules['gate'],
                                      modules['item_embedding'], getattr(self, name + '_original_adj'))

    def encode_image_teacher(self):
        return self._encode_content_teacher('image')

    def encode_text_teacher(self):
        return self._encode_content_teacher('text')

    def fuse_student(self, content_embeds, image_embeds, text_embeds):
        views = [view for view in (image_embeds, text_embeds) if view is not None]
        if not views:
            side = torch.zeros_like(content_embeds)
        elif len(views) == 1:
            side = self.STUDENT_FUSION_COEFF * views[0]
        else:
            weights = self.softmax(torch.cat([self.query_common(view) for view in views], dim=-1))
            redundant = sum(weight.unsqueeze(1) * view for weight, view in zip(weights.unbind(dim=1), views))
            side = self.STUDENT_FUSION_COEFF * (sum(views) - redundant)
        return content_embeds + side, side

    def forward(self, adj=None, train=False, return_views=False):
        content = self.encode_ui_student(adj)
        image, text = self.encode_image_student(), self.encode_text_student()
        fused, side = self.fuse_student(content, image, text)
        users, items = torch.split(fused, (self.n_users, self.n_items))
        if return_views:
            return users, items, side, content, image, text
        if train:
            return users, items, side, content
        return users, items

    def bpr_loss(self, users, pos_items, neg_items):
        margin = (users * (pos_items - neg_items)).sum(dim=-1)
        regularizer = 0.5 * (users.square().sum() + pos_items.square().sum() + neg_items.square().sum())
        return F.softplus(-margin).mean(), self.reg_weight * regularizer / max(users.shape[0], 1), 0.0

    def InfoNCE(self, view1, view2, temperature):
        logits = F.normalize(view1, dim=-1) @ F.normalize(view2, dim=-1).t() / temperature
        return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))

    def _kd_scale(self):
        if self.epoch <= self.kd_warmup_epochs:
            return 0.0
        return min(1.0, (self.epoch - self.kd_warmup_epochs) / max(self.kd_ramp_epochs, 1))

    def _compute_online_expert_views(self):
        return {'ui': self.encode_ui_expert(), 'img': self.encode_image_expert(), 'txt': self.encode_text_expert()}

    @torch.no_grad()
    def _compute_teacher_views(self):
        return {'ui': self.encode_ui_teacher(), 'img': self.encode_image_teacher(), 'txt': self.encode_text_teacher()}

    def _rank_kd_loss_logits(self, users, pos_items, neg_items, stu_user_e, stu_pos_e, stu_neg_e, t_views):
        student_margin = (stu_user_e * (stu_pos_e - stu_neg_e)).sum(dim=-1)
        student_logits = torch.stack((student_margin / self.kd_tau, torch.zeros_like(student_margin)), dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        views = [(name, t_views[name]) for name in self.TEACHER_NAMES if t_views.get(name) is not None]
        if not views:
            raise ValueError('Rank distillation requires at least one available EMA teacher')
        with torch.no_grad():
            target = torch.zeros_like(student_log_probs)
            teacher_count = torch.zeros_like(student_margin)
            for name, view in views:
                teacher_users, teacher_items = torch.split(view, (self.n_users, self.n_items))
                teacher_margin = (teacher_users[users] * (teacher_items[pos_items] - teacher_items[neg_items])).sum(dim=-1)
                supported = (teacher_margin > 0) & (teacher_margin > student_margin.detach())
                teacher_logits = torch.stack((teacher_margin / self.kd_tau, torch.zeros_like(teacher_margin)), dim=-1)
                target.add_(F.softmax(teacher_logits, dim=-1) * supported.unsqueeze(-1))
                teacher_count.add_(supported)
            active = teacher_count > 0
            target.div_(teacher_count.clamp_min(1).unsqueeze(-1))
        per_pair = F.kl_div(student_log_probs, target, reduction='none').sum(dim=-1)

        return (per_pair * active).mean() * self.kd_tau ** 2

    def _cal_teacher_rec_loss(self, users, pos, negs, t_views):
        losses = {}
        for name, view in t_views.items():
            if view is not None:
                user_e, item_e = torch.split(view, (self.n_users, self.n_items))
                losses[name] = self.bpr_loss(user_e[users], item_e[pos], item_e[negs])[0]
        return losses

    def _student_regularization(self):
        regularizer = self.item_id_embedding.weight.new_zeros(())
        for name, parameter in self.named_parameters():
            if not name.startswith(('t_', 'ema_teachers.')):
                regularizer = regularizer + parameter.norm(2).square()
        return regularizer

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        user_e, item_e, _, content, image, text = self.forward(return_views=True)
        loss = self.bpr_loss(user_e[users], item_e[pos_items], item_e[neg_items])[0]
        loss = loss + self.reg_weight * self._student_regularization()
        if self.cl_weight:
            item_nodes = pos_items + self.n_users
            for view in (image, text):
                if view is not None:
                    loss = loss + self.cl_weight * (
                        self.InfoNCE(content[users], view[users], 0.2)
                        + self.InfoNCE(content[item_nodes], view[item_nodes], 0.2))
        if self.use_kd:
            expert_views = self._compute_online_expert_views()
            teacher_losses = self._cal_teacher_rec_loss(users, pos_items, neg_items, expert_views)
            loss = loss + torch.stack(list(teacher_losses.values())).mean()
            scale = self._kd_scale()
            if scale > 0 and self.kd_rank_weight > 0:
                teacher_views = self._compute_teacher_views()
                loss = loss + scale * self.kd_rank_weight * self._rank_kd_loss_logits(
                    users, pos_items, neg_items, user_e[users], item_e[pos_items], item_e[neg_items], teacher_views)
        return loss

    def pre_epoch_processing(self):
        self.clear_eval_cache()
        self.epoch += 1

    def post_optimizer_step(self):
        if self.use_kd:
            self.ema_update_teacher()
        self.clear_eval_cache()

    def clear_eval_cache(self):
        self._eval_cache = None

    @torch.no_grad()
    def prepare_for_eval(self):
        self._eval_cache = self.forward()

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        if self._eval_cache is None:
            self.prepare_for_eval()
        user_embeddings, item_embeddings = self._eval_cache
        return user_embeddings[interaction[0]] @ item_embeddings.t()

