import torch
from torch.optim import Optimizer


class AdaptiveChebyshev(Optimizer):
    def __init__(self, params, warm_up_steps=10, hvp_steps=1, exact_mu=True,
                 my_heuristic_ratio=1e-3, epsilon=1e-6):
        defaults = dict(lr=1.0, warm_up_steps=warm_up_steps,
                        hvp_steps=hvp_steps, exact_mu=exact_mu,
                        my_heuristic_ratio=my_heuristic_ratio, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Шаг оптимизатора
        """
        if closure is None:
            raise RuntimeError('closure must be provided')
        # 1. Сохранение базовых градиентов
        base_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    base_grads[p] = p.grad.clone()
                else:
                    raise RuntimeError('grad must be provided')

        # Вычисление Hessian-Vector Product
        # H(x) * v ≈ (∇f(x + eps*v) - ∇f(x)) / eps
        def compute_hvp(v_dict, eps):
            # Сдвиг весов вдоль v
            for p, v in v_dict.items():
                p.add_(v, alpha=eps)

            with torch.enable_grad():
                closure()
            # Конечные разности градиентов для каждого слоя
            hvp_dict = {}
            for p in base_grads.keys():
                hvp_dict[p] = (p.grad - base_grads[p]) / eps
            # Возврат весов в исходную точку
            for p, v in v_dict.items():
                p.sub_(v, alpha=eps)
            return hvp_dict

        # 2. Инициализация self.state для каждого слоя
        is_first_step = False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    is_first_step = True
                    state['step'] = 0
                    # Генерация случайных начальных направлений
                    v_max = torch.randn_like(p)
                    state['v_max'] = v_max.div_(v_max.norm() + 1e-12)

                    v_min = torch.randn_like(p)
                    state['v_min'] = v_min.div_(v_min.norm() + 1e-12)

                state['step'] += 1

        #извлечение гиперпараметров
        group = self.param_groups[0]
        n_iters = group['warm_up_steps'] if is_first_step else group['hvp_steps']
        eps = group['epsilon']
        exact_mu = group['exact_mu']
        mu_ratio = group['my_heuristic_ratio']

        # 3. Степенной метод для поиска максимального (и минимального) собственных чисел
        for _ in range(n_iters):
            # Плоская карта векторов v_max по всем слоям
            v_max_dict = {p: self.state[p]['v_max'] for group in self.param_groups
                          for p in group['params'] if p.grad is not None}
            y_max_dict = compute_hvp(v_max_dict, eps)
            # Итерация степенного метода для оценки L (lam_max)
            for p, y_max in y_max_dict.items():
                state = self.state[p]
                y_norm = y_max.norm() + 1e-12
                state['v_max'].copy_(y_max / y_norm)
                state['lam_max'] = torch.abs(torch.sum(state['v_max'] * y_max))
            # Если включен честный подсчет mu (минимального собственного числа)
            if exact_mu:
                v_min_dict = {p: self.state[p]['v_min'] for group in self.param_groups
                              for p in group['params'] if p.grad is not None}
                y_min_hvp = compute_hvp(v_min_dict, eps)
                # Спектральный сдвиг: поиск min в поиск max для (L*I - H)
                for p, hvp_v_min in y_min_hvp.items():
                    state = self.state[p]
                    lam_max = state['lam_max']
                    v_min = state['v_min']

                    y_shift = lam_max * v_min - hvp_v_min
                    shift_norm = y_shift.norm() + 1e-12
                    state['v_min'].copy_(y_shift / shift_norm)

                    sigma = torch.abs(torch.sum(state['v_min'] * y_shift))
                    state['lam_min'] = torch.abs(lam_max - sigma)
            else:
                #Эвристика, если точный спектр не так важен
                for p in base_grads.keys():
                    state = self.state[p]
                    state['lam_min'] = state['lam_max'] * mu_ratio
        # 4. Чебышёвский шаг спуска
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                L = state['lam_max']
                mu = state['lam_min']

                eta = 2 / (L + mu + 1e-16)

                # Восстановление сохраненного в начале градиента в исходной точке
                p.grad.copy_(base_grads[p])
                #Шаг против градиента
                p.add_(p.grad, alpha=-eta)

                group['lr'] = eta if isinstance(eta, float) else eta.item()

        return None