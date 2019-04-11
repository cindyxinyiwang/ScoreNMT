import math
import torch
from torch.optim.optimizer import Optimizer


class customAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, hparams, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.hparams = hparams
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(customAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def save_gradients(self, lan_id):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    param_state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                if not "ave_grad" in param_state:
                    param_state["ave_grad"] = [torch.zeros_like(p.data) for _ in range(self.hparams.lan_size)]
                if p.grad is None: continue
                scale_0, scale_1 = torch.FloatTensor([0.15]), torch.FloatTensor([0.85])
                if self.hparams.cuda: 
                  scale_0, scale_1 = scale_0.cuda(), scale_1.cuda()
                d_p = p.grad.data
                param_state["ave_grad"][lan_id] = scale_0*param_state["ave_grad"][lan_id] + scale_1*d_p
                #denom = param_state['exp_avg_sq'].sqrt().add_(group['eps'])
                #param_state["ave_grad"][lan_id] = scale_0*param_state["ave_grad"][lan_id] + scale_1*param_state['exp_avg'] / denom 
    
    def get_cosine_sim(self):
        # return a list of cosine sim of base lan and the lan_id
        cosine_prod = [0 for _ in range(self.hparams.lan_size)]
        cosine_norm = [0 for _ in range(self.hparams.lan_size)]
        base_lan_id = self.hparams.base_lan_id
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if not "ave_grad" in param_state:
                    param_state["ave_grad"] = [torch.zeros_like(p.data) for _ in range(self.hparams.lan_size)]
                if p.grad is None: continue
                for i in range(self.hparams.lan_size):
                  prod = param_state["ave_grad"][i] * param_state["ave_grad"][base_lan_id]
                  prod = prod.sum()
                  norm = param_state["ave_grad"][i].norm(2) ** 2
                  cosine_prod[i] = cosine_prod[i] + prod  
                  cosine_norm[i] = cosine_norm[i] + norm  
        cosine_dist = [p / (n.sqrt()*cosine_norm[base_lan_id].sqrt() +1e-10) for p, n in zip(cosine_prod, cosine_norm)]
        return cosine_dist

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
