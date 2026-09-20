import torch


class SAMOptimizer:
    def __init__(
        self,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        epsilon=1e-12,
    ):
        if rho < 0:
            raise ValueError(
                "rho must be non-negative."
            )

        self.base_optimizer = (
            base_optimizer
        )

        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.epsilon = float(epsilon)

        self._perturbations = {}

    @property
    def param_groups(self):
        return (
            self.base_optimizer.param_groups
        )

    def zero_grad(
        self,
        set_to_none=True,
    ):
        self.base_optimizer.zero_grad(
            set_to_none=set_to_none
        )

    def _parameters_with_gradients(
        self,
    ):
        return [
            parameter
            for parameter_group
            in self.param_groups
            for parameter
            in parameter_group["params"]
            if parameter.grad is not None
        ]

    def gradient_norm(self):
        parameters = (
            self._parameters_with_gradients()
        )

        if not parameters:
            raise RuntimeError(
                "SAM received no gradients."
            )

        gradient_norms = []

        for parameter in parameters:
            if self.adaptive:
                scaled_gradient = (
                    parameter.detach().abs()
                    * parameter.grad
                )
            else:
                scaled_gradient = (
                    parameter.grad
                )

            gradient_norms.append(
                scaled_gradient.norm(
                    p=2
                )
            )

        return torch.stack(
            gradient_norms
        ).norm(
            p=2
        )

    @torch.no_grad()
    def first_step(self):
        gradient_norm = (
            self.gradient_norm()
        )

        scale = (
            self.rho
            / (
                gradient_norm
                + self.epsilon
            )
        )

        self._perturbations = {}

        for parameter_group in (
            self.param_groups
        ):
            for parameter in (
                parameter_group["params"]
            ):
                if parameter.grad is None:
                    continue

                if self.adaptive:
                    parameter_scale = (
                        parameter.detach()
                        .pow(2)
                    )
                else:
                    parameter_scale = 1.0

                perturbation = (
                    parameter_scale
                    * parameter.grad
                    * scale
                )

                parameter.add_(
                    perturbation
                )

                self._perturbations[
                    parameter
                ] = perturbation

        return gradient_norm.detach()

    @torch.no_grad()
    def restore_parameters(self):
        for parameter, perturbation in (
            self._perturbations.items()
        ):
            parameter.sub_(
                perturbation
            )

        self._perturbations = {}

    @torch.no_grad()
    def second_step(self):
        self.restore_parameters()

        self.base_optimizer.step()

    def state_dict(self):
        return {
            "base_optimizer": (
                self.base_optimizer
                .state_dict()
            ),
            "rho": self.rho,
            "adaptive": self.adaptive,
            "epsilon": self.epsilon,
        }

    def load_state_dict(
        self,
        state_dict,
    ):
        self.base_optimizer.load_state_dict(
            state_dict[
                "base_optimizer"
            ]
        )

        self.rho = float(
            state_dict["rho"]
        )

        self.adaptive = bool(
            state_dict["adaptive"]
        )

        self.epsilon = float(
            state_dict["epsilon"]
        )
