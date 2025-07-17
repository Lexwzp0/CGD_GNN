import torch
import torch.nn as nn
from src.utils.pyg_dataToGraph import DataToGraph
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from src.utils.losses import normal_kl, discretized_gaussian_log_likelihood # Assuming these are available

#%%
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

#%%
def extract(a, t, x_shape):
    """
    Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape.
    """
    bs = t.shape[0]
    # Ensure a and t are on the same device
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(bs, *((1,) * (len(x_shape) - 1)))

#%%

class GuidedGaussianDiffusion(nn.Module):
    def __init__(
        self,
        num_steps,
        classifier_scale = 1.0,
        ):
        super().__init__()
        self.num_steps = num_steps
        self.classifier_scale = classifier_scale
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 默认使用 Improved DDPM 的 beta 调度
        betas = self._cosine_beta_schedule_with_offset(
            num_steps=num_steps,
            max_beta=0.999,
            offset=0.008,
            dtype=torch.float64
        )

        # Calculate alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64, device=alphas.device), alphas_cumprod[:-1]], 0)

        # Convert all variables to float32
        self.betas = betas.to(torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(torch.float32)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(torch.float32)

        # Calculations for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Calculations for p(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas).to(torch.float32) / (1. - self.alphas_cumprod)
        )

    def _cosine_beta_schedule_with_offset(self, num_steps, max_beta=0.999, offset=0.008, dtype=torch.float64):
        """Core function for Improved DDPM cosine schedule"""
        t = torch.linspace(0, 1, num_steps + 1)
        alpha_bars = torch.cos((t + offset) / (1 + offset) * torch.pi / 2) ** 2

        betas = []
        for i in range(num_steps):
            beta = 1 - (alpha_bars[i+1] / alpha_bars[i])
            betas.append(min(beta.item(), max_beta))

        return torch.tensor(betas, dtype=dtype)

    # MODIFIED: Renamed function to be clear about its purpose
    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Calculates predicted x_0 from x_t and predicted noise eps.
        x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, batch_labels = None, clip_denoised = True
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, batch_labels, clip_denoised=clip_denoised
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def cond_fn(self,classifier, x, t, y):
        """Classifier gradient calculation function"""
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale

    def condition_mean(self, classifier, p_mean_var, x, t, batch_y):
        gradient = self.cond_fn(classifier, x, t, batch_y)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # MODIFIED: This function is now completely changed for epsilon prediction.
    def p_mean_variance(
            self, model, x, t, batch_labels = None, clip_denoised=True, denoised_fn=None,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # The model's output now has two parts: predicted epsilon and variance values
        model_output_full = model(x, t, batch_labels)
        assert model_output_full.shape == (B, C * 2, *x.shape[2:])
        model_eps, model_var_values = torch.split(model_output_full, C, dim=1)

        # Learned variance calculation (this part is the same)
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        # NEW: The core logic for epsilon prediction
        # 1. Predict x_0 from the model's predicted noise (epsilon).
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_eps)

        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        # 2. Use the predicted x_0 to calculate the mean of p(x_{t-1}|x_t)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self,
                 model,
                 classifier,
                 x,
                 t,
                 batch_labels,
                 clip_denoised = True,
                 denoised_fn = None,
                 ):
        # This function works without changes because it depends on p_mean_variance
        out = self.p_mean_variance(model, x, t, batch_labels, clip_denoised, denoised_fn)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        # Apply guidance
        out["mean"] = self.condition_mean(
             classifier, out, x, t, batch_labels
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    # These sampling loops work without changes
    def p_sample_loop_progressive(
            self, model, classifier, shape, batch_labels,
            noise = None, clip_denoised=True, denoised_fn=None, progress=False,
    ):
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=self.device)
        indices = list(range(self.num_steps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.p_sample(
                    model, classifier, img, t, batch_labels,
                    clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
            self, model, classifier, shape, batch_labels,
            noise=None, clip_denoised=True, denoised_fn=None, progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model, classifier, shape, batch_labels,
            noise = noise, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, progress=progress,
        ):
            final = sample
        return final["sample"]

    # MODIFIED: The main loss function is now updated for epsilon prediction.
    def training_losses(self, model, x_start, t, batch_labels, noise=None):
        if noise is None:
            # The target for the model is the noise we add
            noise = torch.randn_like(x_start)
        
        # Get the noised image x_t
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        
        # Get the model's prediction
        model_output_full = model(x_t, t, batch_labels)

        B, C = x_t.shape[:2]
        assert model_output_full.shape == (B, C * 2, *x_t.shape[2:])

        # The first half of the output is the predicted noise (epsilon)
        model_eps, model_var_values = torch.split(model_output_full, C, dim=1)
        
        # NEW: The main MSE loss is now on the noise
        # The model is trained to predict the original noise
        terms["mse"] = mean_flat((noise - model_eps) ** 2)
        
        # The VLB term provides the loss for the learned variance
        # It uses the epsilon prediction internally via p_mean_variance
        # We use a small weight to prevent instability, as in Improved DDPM
        vb_weight = 0.001
        terms["vb"] = self._vb_terms_bpd(
            model=model,
            x_start=x_start,
            x_t=x_t,
            t=t,
            batch_labels=batch_labels,
            clip_denoised=False
        )["output"]
        
        # The final loss is a hybrid of the simple noise prediction loss and the VLB
        terms["loss"] = terms["mse"] + vb_weight * terms["vb"]
        
        # Safety check for non-finite values
        if not torch.isfinite(terms["loss"]).all():
            print("Warning: Non-finite loss value! Replacing with a safe value.")
            # Set loss to a safe, non-zero value to avoid breaking the optimizer
            safe_loss = torch.full_like(terms["loss"], 0.1, requires_grad=True)
            terms["loss"] = torch.where(
                torch.isfinite(terms["loss"]),
                terms["loss"],
                safe_loss
            )
        
        return terms