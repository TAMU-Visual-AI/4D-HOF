"""Inference samplers extracted from the research Brownian bridge implementation."""

import torch
import torch.nn as nn
import enum
import math


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelType(enum.Enum):
    VELOCITY = enum.auto()   # model predicts v(x_t, t)  ≈ u_t
    X1       = enum.auto()   # model predicts x1 directly


# ---------------------------------------------------------------------------
# Brownian Bridge Path
# ---------------------------------------------------------------------------

class BrownianBridgePath:
    """
    Stochastic interpolant between x0 and x1 via a Brownian Bridge.

    Args:
        beta: controls the width of the bridge noise; set to 0 for
              a deterministic (linear flow) path.
    """

    def __init__(self, beta: float = 0.001):
        self.beta = beta

    def mean(self, t, x0, x1):
        """mu(t) = (1-t)*x0 + t*x1"""
        t_ = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - t_) * x0 + t_ * x1

    def std(self, t):
        """sigma(t) = beta * sqrt(t*(1-t))"""
        return self.beta * torch.sqrt(t * (1 - t)).clamp(min=1e-6)


    def sample_xt(self, t, x0, x1):
        """
        Draw a bridge sample x_t ~ p(x_t | x0, x1).
        Returns (x_t, eps) where eps is the noise used.
        """
        mu    = self.mean(t, x0, x1)
        sigma = self.std(t).view(-1, *([1] * (x0.dim() - 1)))
        eps   = torch.randn_like(x0)
        x_t   = mu + sigma * eps
        return x_t, eps


    def velocity(self, t, x0, x1, x_t):
        """
        Analytical velocity u_t = d/dt E[x_t | x0, x1] corrected for noise.

        u_t = (x1 - x0) + d_sigma/dt * eps
            = (x1 - x0) + (1 - 2t) / (2*sigma(t)) * (x_t - mu(t))

        This is the target for velocity prediction.
        """
        t_    = t.view(-1, *([1] * (x0.dim() - 1)))
        mu    = self.mean(t, x0, x1)
        sigma = self.std(t).view(-1, *([1] * (x0.dim() - 1)))

        # d(sigma)/dt = beta * (1 - 2t) / (2 * sqrt(t*(1-t)))
        #             = (1 - 2t) / (2 * sigma / beta^2) ... simplified:
        #             = sigma * (1 - 2*t) / (2 * t * (1 - t))  [via quotient rule]
        dsigma_dt = sigma * (1 - 2 * t_) / (2 * t_.clamp(min=1e-6) * (1 - t_).clamp(min=1e-6))

        # residual noise component
        noise_contribution = dsigma_dt * (x_t - mu) / sigma.clamp(min=1e-6)
        return (x1 - x0) + noise_contribution


# ---------------------------------------------------------------------------
# Bridge Matching Transport
# ---------------------------------------------------------------------------

class BridgeMatching:
    """
    Core training and sampling framework for Brownian Bridge Matching.

    Args:
        model_type: VELOCITY or X1
        beta:       bridge noise scale (0 = deterministic linear path)
        train_eps:  time margin during training  (avoid t=0,1 numerics)
        sample_eps: time margin during sampling
        num_steps:  if set, enables discrete few-step mode.
                    Training samples t uniformly from a fixed grid of
                    num_steps-1 interior points; inference uses
                    sample_few_steps() which steps exactly on that grid.
                    If None, continuous-time mode is used (default).
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.VELOCITY,
        beta: float = 1e-3,
        train_eps: float = 1e-3,
        sample_eps: float = 1e-3,
        num_steps: int = None,
    ):
        self.model_type  = model_type
        self.path        = BrownianBridgePath(beta=beta)
        self.train_eps   = train_eps
        self.sample_eps  = sample_eps
        self.num_steps   = num_steps

        # Pre-build the discrete timestep grid if in few-step mode.
        # Grid: num_steps+1 boundaries → num_steps-1 interior points used for training.
        # Example for num_steps=4:  boundaries [0, 0.25, 0.5, 0.75, 1.0]
        #                           training t: [0.25, 0.50, 0.75]
        if num_steps is not None:
            boundaries = torch.linspace(0.0, 1.0, num_steps + 1)  # [0, 1/N, ..., 1]
            self._train_grid  = boundaries[1:-1]   # interior points only (exclude 0 and 1)
            self._sample_grid = boundaries         # full grid used during inference


    # ------------------------------------------------------------------
    # Drift functions for ODE / SDE samplers
    # ------------------------------------------------------------------

    def get_drift(self):
        """
        Returns the drift function dx_t/dt used by ODE samplers.

        For velocity models: drift = model_out directly.
        For x1 models:       drift = velocity computed from predicted x1.
        """
        path = self.path

        if self.model_type == ModelType.VELOCITY:
            def drift_fn(x_t, t, x0, model, **kwargs):
                return model(x_t, t, **kwargs)

        elif self.model_type == ModelType.X1:
            def drift_fn(x_t, t, x0, model, **kwargs):
                x1_pred = model(x_t, t, **kwargs)
                return path.velocity(t, x0, x1_pred, x_t)

        return drift_fn

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    def sample_ode(
        self,
        x0,
        model,
        num_steps=50,
        method="dopri5",
        atol=1e-5,
        rtol=1e-3,
        model_kwargs=None,
    ):
        """
        ODE sampler using torchdiffeq (adaptive solvers available).

        Args:
            x0:         source poses [B, ..., D]
            model:      trained neural network
            num_steps:  number of timepoints saved in the trajectory
            method:     torchdiffeq solver — 'dopri5' (adaptive, recommended),
                        'euler', 'midpoint', 'rk4', 'heun3', etc.
            atol:       absolute tolerance (adaptive solvers only)
            rtol:       relative tolerance (adaptive solvers only)
            model_kwargs: extra kwargs for model (e.g. conditioning)

        Returns:
            trajectory: list of tensors length = num_steps+1,
                        from t=sample_eps to t=1-sample_eps
        """
        from torchdiffeq import odeint

        if model_kwargs is None:
            model_kwargs = {}

        drift_fn = self.get_drift()
        t0       = self.sample_eps
        t1       = 1.0 - self.sample_eps

        # torchdiffeq expects f(t, x) with scalar t and flat/tensor x
        # We store x0 in closure so drift_fn can use it
        def ode_func(t, x):
            t_batch = t.expand(x.shape[0]).to(x)
            return drift_fn(x, t_batch, x0, model, **model_kwargs)

        t_eval = torch.linspace(t0, t1, num_steps + 1, device=x0.device, dtype=x0.dtype)

        # odeint returns [num_steps+1, B, ..., D]
        traj = odeint(
            ode_func,
            x0,
            t_eval,
            method=method,
            atol=atol,
            rtol=rtol,
        )

        # Return as a list of [B, ..., D] tensors to match sample_sde interface
        return list(traj)


    def sample_few_steps(self, x0, model, model_kwargs=None):
        """
        Discrete few-step inference that mirrors the training grid exactly.
        Only available when num_steps was set at construction time.

        The sampler walks the full grid [t_0=0, t_1, ..., t_N=1] using
        fixed Euler steps of size dt = 1/num_steps.  At each interior
        step the model is queried at the current grid point and the state
        is advanced to the next boundary.

        For X1 models an optional bridge re-noising is applied between
        steps (similar to DDIM-style ancestral sampling): the predicted x1
        is used to re-sample x_{t+1} from the bridge rather than blindly
        stepping the ODE.  This keeps the trajectory on the bridge manifold
        and tends to give sharper results with very few steps.

        Args:
            x0:           source poses [B, ..., D]
            model:        trained neural network
            model_kwargs: extra kwargs for model (e.g. conditioning)

        Returns:
            trajectory: list of tensors of length num_steps+1,
                        one entry per grid boundary [t_0 ... t_N]
        """
        assert self.num_steps is not None, \
            "sample_few_steps() requires num_steps to be set at construction."

        if model_kwargs is None:
            model_kwargs = {}

        grid     = self._sample_grid.to(x0.device, dtype=x0.dtype)  # [N+1]
        drift_fn = self.get_drift()

        x    = x0.clone()
        traj = [x.clone()]

        for i in range(len(grid) - 1):           # iterate over N intervals
            t_cur  = grid[i]
            t_next = grid[i + 1]
            dt     = t_next - t_cur

            t_batch = t_cur.expand(x.shape[0])

            if self.model_type == ModelType.X1:
                # Bridge re-noising: predict x1, then re-sample from the
                # bridge at t_next instead of using a plain Euler step.
                # At the last step (t_next == 1) just return the prediction.
                x1_pred = model(x, t_batch, **model_kwargs)
                if i == len(grid) - 2:           # final step → return prediction
                    x = x1_pred
                else:
                    x, _ = self.path.sample_xt(t_next.expand(x.shape[0]), x0, x1_pred)

            else:  # VELOCITY: plain Euler step
                dx = drift_fn(x, t_batch, x0, model, **model_kwargs)
                x  = x + dx * dt

            traj.append(x.clone())

        return traj

    # ------------------------------------------------------------------
    # Guided Samplers (classifier-guidance style)
    # ------------------------------------------------------------------

    def sample_ode_guided(
        self,
        x0,
        model,
        guidance_fn,
        guidance_scale=1.0,
        num_steps=50,
        grad_clip_norm=1.0,
        guidance_start_t=0.0,
        guidance_iters=1,
        model_kwargs=None,
    ):
        """
        Euler ODE sampler with per-step gradient-based guidance.

        Unlike sample_ode() which uses torchdiffeq adaptive solvers, this
        uses a fixed-step Euler loop so that we can inject guidance gradients
        at every step.

        At each step:
            drift  = model velocity / derived velocity
            grad   = ∇_{x_t} guidance_loss(x_t, t, ...)
            x_{t+dt} = x_t + drift * dt - guidance_scale * grad

        Args:
            x0:             source poses [B, ..., D]
            model:          trained neural network
            guidance_fn:    callable(x_t, t_scalar, model_kwargs) -> gradient tensor
                            same shape as x_t.  Should return the *gradient*
                            of the guidance loss w.r.t. x_t (already computed
                            outside this function).
            guidance_scale: scalar multiplier for the guidance gradient.
            num_steps:      number of fixed Euler steps.
            grad_clip_norm: max L2 norm for gradient clipping (stabilisation).
                            Set to None to disable.
            guidance_start_t: timestep ratio in [0, 1] at which guidance begins.
                            Guidance is only applied when t >= guidance_start_t.
                            Default 0.0 applies guidance at every step.
            model_kwargs:   extra kwargs for model (e.g. conditioning).

        Returns:
            trajectory: list of tensors, length = num_steps+1.
        """
        if model_kwargs is None:
            model_kwargs = {}

        drift_fn = self.get_drift()
        dt       = (1.0 - 2 * self.sample_eps) / num_steps
        t0       = self.sample_eps

        x    = x0.clone()
        traj = [x.clone()]

        for i in range(num_steps):
            t_val   = t0 + i * dt
            t_batch = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)

            # --- deterministic drift (no grad needed) ---
            with torch.no_grad():
                dx = drift_fn(x, t_batch, x0, model, **model_kwargs)

            x = x + dx * dt

            # --- guidance gradient (only after guidance_start_t) ---
            if t_val >= guidance_start_t:
                for _ in range(guidance_iters):
                    grad = guidance_fn(x, t_val, model_kwargs)

                    if grad_clip_norm is not None:
                        grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                        grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                        clip_coef = (grad_clip_norm / grad_norm).clamp(max=1.0)
                        grad = grad * clip_coef

                    x = x - guidance_scale * grad

            traj.append(x.clone())

        return traj

    def sample_few_steps_guided(
        self,
        x0,
        model,
        guidance_fn,
        guidance_scale=1.0,
        grad_clip_norm=1.0,
        guidance_start_t=0.0,
        guidance_iters=1,
        model_kwargs=None,
        guide_final_prediction=False,
    ):
        """
        Discrete few-step inference with per-step gradient-based guidance.

        Mirrors sample_few_steps() but adds a guidance gradient correction
        after each Euler / re-noising step.

        Args:
            x0:             source poses [B, ..., D]
            model:          trained neural network
            guidance_fn:    callable(x_t, t_scalar, model_kwargs) -> gradient tensor
            guidance_scale: scalar multiplier for the guidance gradient.
            grad_clip_norm: max L2 norm for gradient clipping.
            guidance_start_t: timestep ratio in [0, 1] at which guidance begins.
                            Default 0.0 applies guidance at every step.
            model_kwargs:   extra kwargs for model (e.g. conditioning).
            guide_final_prediction: guide the final sample at t_next, including
                            clean X1 predictions and late guidance start times.

        Returns:
            trajectory: list of tensors, length = num_steps+1.
        """
        assert self.num_steps is not None, \
            "sample_few_steps_guided() requires num_steps to be set at construction."

        if model_kwargs is None:
            model_kwargs = {}

        grid     = self._sample_grid.to(x0.device, dtype=x0.dtype)
        drift_fn = self.get_drift()

        x    = x0.clone()
        traj = [x.clone()]

        for i in range(len(grid) - 1):
            t_cur  = grid[i]
            t_next = grid[i + 1]
            dt     = t_next - t_cur

            t_batch = t_cur.expand(x.shape[0])

            with torch.no_grad():
                if self.model_type == ModelType.X1:
                    x1_pred = model(x, t_batch, **model_kwargs)
                    if i == len(grid) - 2:
                        x = x1_pred
                    else:
                        x, _ = self.path.sample_xt(t_next.expand(x.shape[0]), x0, x1_pred)
                else:  # VELOCITY
                    dx = drift_fn(x, t_batch, x0, model, **model_kwargs)
                    x  = x + dx * dt

            # A clean final X1 prediction otherwise overwrites earlier guidance.
            final_step = i == len(grid) - 2
            guidance_t = t_next.item() if final_step and guide_final_prediction else t_cur.item()
            apply_guidance = guidance_t >= guidance_start_t
            if self.model_type == ModelType.X1 and final_step and not guide_final_prediction:
                apply_guidance = False

            if apply_guidance:
                for _ in range(guidance_iters):
                    grad = guidance_fn(x, guidance_t, model_kwargs)

                    if grad_clip_norm is not None:
                        grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                        grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                        clip_coef = (grad_clip_norm / grad_norm).clamp(max=1.0)
                        grad = grad * clip_coef

                    x = x - guidance_scale * grad

            traj.append(x.clone())

        return traj


def create_bridge(
    prediction="velocity",
    beta=0.001,
    num_steps=None,
    train_eps=None,
    sample_eps=None,
) -> BridgeMatching:
    """
    Factory function for creating a BridgeMatching object.

    Args:
        prediction:     model prediction type; one of:
                          'velocity' (default) — model predicts v(x_t, t)
                          'x1'                 — model predicts clean x1 directly
        beta:           bridge noise scale; controls stochasticity of the bridge.
                          0.0  → deterministic linear interpolation (flow matching)
                          0.1  → recommended default for normalized 3D poses
                          1.0  → strong stochasticity (usually too large for poses)
        num_steps:      if set, enables discrete few-step mode. The model is
                        trained and inferred on a fixed uniform grid of N steps.
                        Use sample_few_steps() for inference in this mode.
                        If None, continuous-time mode is used.
        train_eps:      small time margin at t=0 and t=1 during training to
                        avoid numerical issues. Defaults:
                          velocity prediction → 0.0  (stable everywhere)
                          x1 prediction       → 1e-3 (numerically safer)
        sample_eps:     small time margin during sampling. Defaults mirror
                        train_eps when not set.

    Returns:
        BridgeMatching instance ready for training and sampling.

    Examples:
        # Simple continuous velocity model
        bm = create_bridge()

        # X1 prediction with auxiliary x1 loss disabled (redundant but valid)
        bm = create_bridge(prediction='x1')

        # Velocity model with auxiliary x1 loss and soft bridge noise
        bm = create_bridge(prediction='velocity', beta=0.1, x1_loss_weight=0.5)

        # 4-step discrete model
        bm = create_bridge(num_steps=4, prediction='velocity', x1_loss_weight=0.5)

        # Deterministic linear bridge (no stochasticity)
        bm = create_bridge(beta=0.0)
    """

    # --- prediction type ---
    if prediction == "x1":
        model_type = ModelType.X1
    else:
        model_type = ModelType.VELOCITY   # default

    # --- train / sample eps defaults ---
    # velocity prediction is stable at t=0 and t=1 because the bridge
    # velocity is well-defined at the endpoints (no sigma singularity).
    # x1 prediction involves dividing by sigma(t) implicitly so we
    # add a small margin to avoid t=0,1.
    if model_type == ModelType.VELOCITY:
        default_eps = 0.0
    else:
        default_eps = 1e-3

    train_eps  = default_eps if train_eps  is None else train_eps
    sample_eps = default_eps if sample_eps is None else sample_eps

    return BridgeMatching(
        model_type     = model_type,
        beta           = beta,
        train_eps      = train_eps,
        sample_eps     = sample_eps,
        num_steps      = num_steps
    )
