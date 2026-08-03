"""ArcFace angular-margin loss + a timm training task, for open-set-friendly features.

ArcFace (additive angular margin softmax) replaces the plain linear-softmax head with a
normalized-feature / normalized-weight cosine classifier and adds an angular margin `m` to the
ground-truth class before scaling by `s`. This tightens intra-class angles and widens inter-class
angles, producing compact, well-separated clusters on a hypersphere — exactly the geometry the
prototype + cosine-distance open-set method (``nptools/openset.py``) relies on.

Two pieces:

- :class:`ArcFaceLoss` — a loss module that OWNS the class-weight matrix and an inner
  classification criterion (so it composes with label smoothing / soft mixup targets).
- :class:`ArcFaceTask` — a timm :class:`~timm.task.TrainingTask` that wires the loss into
  ``train.py``. The backbone keeps its normal classifier, so the saved checkpoint is a plain timm
  ``state_dict`` that ``openset.py`` loads unchanged; the ArcFace head is training-only.

Enable from ``train.py`` with ``--arcface`` (see ``--arcface-s`` / ``--arcface-m``).
"""
import math
import logging
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.task import TrainingTask

_logger = logging.getLogger(__name__)


class ArcFaceLoss(nn.Module):
    """Additive angular margin softmax loss (ArcFace).

    Owns the class-weight matrix ``W`` of shape ``[num_classes, in_features]``. Given input
    features it computes the cosine similarity between L2-normalized features and L2-normalized
    class weights, adds the angular margin `m` to the ground-truth class, scales by `s`, and
    applies an inner cross-entropy criterion.

    The head is training-only: at inference ``openset.py`` uses the backbone's pre-logits features
    directly and discards this head.

    Math / geometry (why this beats plain linear-softmax for open-set):
        Plain softmax uses the raw dot product as the logit:
            logit_j = W_j . x = ||W_j|| * ||x|| * cos(θ)
        It separates the training classes but does not control the geometry, so same-class
        embeddings can be spread out and different classes can sit close together -- bad when the
        decision rule at test time is cosine distance to a gallery.

        ArcFace strips everything except the ANGLE and enforces a margin around it:
          1. L2-normalize x and every W_j  => ||x||=||W_j||=1, so logit_j = cos(θ_j). Magnitude
             is gone; each W_j is a learned prototype on the unit hypersphere and classification is
             "which prototype is my embedding closest to in angle".
          2. Add an angular margin m to the GROUND-TRUTH class only, before scaling by s:
                 target class : s * cos(θ_y + m)     # penalized -- looks worse on purpose
                 other classes: s * cos(θ_j)          # unchanged
             Since cos is decreasing, cos(θ_y + m) < cos(θ_y): the correct class is made to
             look worse during training, so to drive the loss down the network must pull θ_y
             even smaller -- clustering same-class embeddings tighter and opening a gap of size m to
             other classes. The margin lives in ANGLE space (additive), which is what makes ArcFace
             cleaner than CosFace (additive cosine margin) or SphereFace (multiplicative).

        Hyperparameters: s (scale/radius, ~30-64) lets softmax produce sharp gradients since cosines
        live in [-1, 1]; m (margin, ~0.3-0.5) trades cluster tightness against training stability.

    This compact, well-separated hypersphere geometry is exactly what the prototype + cosine-distance
    open-set method in ``openset.py`` relies on: enrolling a new class = averaging a few normalized
    embeddings into a prototype (no retraining), and an "unknown" cosine threshold cleanly rejects
    out-of-gallery items because same-class similarities stay high and cross-class stay low.

    Args:
        in_features: embedding dimension of the input features
        num_classes: number of classes (equals the class-map line count)
        s: feature scale applied to the cosine logits
        m: angular margin in radians added to the ground-truth class
        base_criterion: inner classification loss over the scaled logits (defaults to
            ``nn.CrossEntropyLoss``); pass ``LabelSmoothingCrossEntropy`` /
            ``SoftTargetCrossEntropy`` to compose with those.
    """

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            s: float = 30.0,
            m: float = 0.50,
            base_criterion: Optional[Union[nn.Module, Callable]] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.base_criterion = base_criterion if base_criterion is not None else nn.CrossEntropyLoss()

        # Precompute margin constants from the fixed m so the forward pass never calls arccos/cos.
        # _cos_m / _sin_m feed the angle-addition identity cos(θ + m) = cos.cos_m - sin.sin_m.
        # _th / _mm are the "easy margin" monotonicity guard (see _margin_logits).
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)          # cos(θ) below this => θ + m > pi
        self._mm = math.sin(math.pi - m) * m      # linear fallback keeps the target logit monotone

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Plain scaled cosine logits (no margin) — used for eval / accuracy.

        Accuracy is measured on the UN-penalized cosines: the margin is a training-time trick, not
        the real decision rule, so metrics use s*cos(θ) directly.
        """
        # F.normalize -> unit vectors; F.linear(a, b) = a @ b.T -> cos(θ) for every sample x class.
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine * self.s

    def _margin_logits(self, cosine: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # sin(θ) from cos(θ) via the Pythagorean identity; clamp guards sqrt against fp noise.
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_(0.0, 1.0))
        # Angle-addition identity: the penalized target logit cos(θ + m), no arccos needed.
        phi = cosine * self._cos_m - sine * self._sin_m                  # cos(θ + m)
        # Monotonicity guard: cos(θ + m) is only a valid (decreasing) penalty while θ + m <= pi.
        # Once θ is so large that θ + m overshoots pi, cos starts rising again and would reward
        # being wrong; below the threshold _th we swap in a linear fallback that stays monotonic. Mostly
        # fires only early in training when features are still near-random.
        phi = torch.where(cosine > self._th, phi, cosine - self._mm)     # keep monotonic near pi
        # Apply the margin to the ground-truth class ONLY: phi where one-hot==1, plain cosine elsewhere.
        one_hot = F.one_hot(target, self.num_classes).to(cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the ArcFace loss.

        Hard integer labels get the angular margin; soft / mixup targets (float, or 2-D one-hot)
        fall back to plain scaled cosine (the margin needs a single ground-truth class index).
        """
        # cos(θ) between unit-normalized embeddings and unit-normalized class prototypes.
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        # The additive angular margin needs a single ground-truth index (the one-hot mask), so it only
        # applies to hard integer labels. Soft / mixup targets (float or 2-D one-hot) have no single
        # correct class -> fall back to plain scaled cosine and let base_criterion handle the mixing.
        if target.ndim == 1 and not torch.is_floating_point(target):
            logits = self._margin_logits(cosine, target)
        else:
            logits = cosine * self.s
        return self.base_criterion(logits, target)


class ArcFaceTask(TrainingTask):
    """timm training task that trains a backbone with an :class:`ArcFaceLoss` head.

    The backbone (``trainable_module``) keeps its normal classifier, so the checkpoint written by
    ``train.py`` is a plain timm ``state_dict`` that ``openset.py`` loads unchanged. The ArcFace
    head lives on this task (via the criterion), so it is NOT part of the model ``state_dict``; it
    is saved separately under ``task_state`` for clean resume.

    Forward extracts pre-logits features, applies the margin head for the loss, and also returns
    the no-margin cosine logits as ``output`` for training-time metrics.

    Notes:
        - The backbone's own softmax classifier is NOT trained under ArcFace (the margin head
          replaces it). Consume the checkpoint via ``openset.py`` (cosine metric), not as a raw
          softmax classifier.
        - Intended for single-process / DataParallel training. Under DDP the head's gradients are
          not synchronized across ranks (it is owned by the task, not the DDP-wrapped module).
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: ArcFaceLoss,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)
        self.trainable_module = model
        self.criterion = criterion  # ArcFaceLoss owns the head weight (training-only)

        # Announce on stdout so it is obvious ArcFace is active for this run (logging config
        # may route _logger elsewhere or be silenced).
        print(
            f">>> ArcFace loss IN USE: in_features={criterion.in_features} "
            f"num_classes={criterion.num_classes} s={criterion.s} m={criterion.m}",
            flush=True,
        )

        if self.verbose:
            _logger.info(
                f"ArcFaceTask: in_features={criterion.in_features} num_classes={criterion.num_classes} "
                f"s={criterion.s} m={criterion.m}"
            )

    @property
    def head(self) -> ArcFaceLoss:
        """The ArcFace head/criterion (exposed for the eval path in train.py)."""
        return self.criterion

    def _backbone(self) -> nn.Module:
        """Unwrap DDP / EMA wrappers to reach ``forward_features`` / ``forward_head``."""
        m = self.trainable_module
        return m.module if hasattr(m, 'module') else m

    def prepare_distributed(
            self,
            device_ids: Optional[list] = None,
            **ddp_kwargs,
    ) -> 'ArcFaceTask':
        from torch.nn.parallel import DistributedDataParallel as DDP
        _logger.warning(
            "ArcFaceTask under DDP: the ArcFace head gradients are not synchronized across ranks; "
            "prefer single-process training."
        )
        self.trainable_module = DDP(self.trainable_module, device_ids=device_ids, **ddp_kwargs)
        return self

    def compile(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> nn.Module:
        self.trainable_module = torch.compile(self.trainable_module, backend=backend, mode=mode, **compile_kwargs)
        self.eval_model = self.trainable_module
        return self.trainable_module

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        net = self._backbone()
        # pre_logits=True grabs the embedding BEFORE the backbone's own softmax classifier -- this is
        # the vector ArcFace normalizes and the exact one openset.py uses at inference. The backbone's
        # built-in classifier is never trained under ArcFace; the margin head replaces it.
        feat = net.forward_head(net.forward_features(input), pre_logits=True)   # [B, D]
        loss = self.criterion(feat, target)
        output = self.criterion.logits(feat)          # no-margin cosine logits for train metrics
        return {
            'loss': loss,
            'output': output,
        }

    def get_task_state(self, module: Optional[nn.Module] = None, ema: bool = False) -> Dict[str, torch.Tensor]:
        """Save the training-only ArcFace head under task_state (no EMA copy)."""
        if ema:
            return {}
        return {'arcface': self.criterion.state_dict()}

    def load_task_state(
            self,
            state: Optional[Dict[str, torch.Tensor]],
            strict: bool = True,
            module: Optional[nn.Module] = None,
            ema: bool = False,
    ) -> None:
        if ema or not state:
            return
        head_state = state.get('arcface')
        if head_state:
            self.criterion.load_state_dict(head_state, strict=strict)
