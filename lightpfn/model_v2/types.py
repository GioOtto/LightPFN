"""Internal, typed S0 inputs. This is not the public sklearn API."""

from dataclasses import dataclass, replace

import torch


@dataclass
class Table:
    """Separate numerical storage and exact int32 nominal identity.

    Arrays are (batch, rows, columns); categorical_mask and column_uid are
    (batch, columns), row_uid is (batch, rows). Missing values use an explicit
    mask (and numerical NaNs). UIDs are stable metadata, never target-derived.
    Feature order is canonicalized by column_uid before circular grouping/RoPE.
    Unused entries in either value array are ignored, including NaNs.
    n_columns (optional, (batch,) int64) pads tables of different widths into
    one batch: each task has n_columns real columns, and its other columns,
    those with the largest UIDs, are padding that no real cell ever sees.
    """

    numerical: torch.Tensor
    categorical: torch.Tensor
    categorical_mask: torch.Tensor
    missing: torch.Tensor
    row_uid: torch.Tensor
    column_uid: torch.Tensor
    n_columns: torch.Tensor | None = None

    def validate(self):
        x = self.numerical
        if x.ndim != 3 or not x.is_floating_point() or min(x.shape) < 1:
            raise ValueError("numerical must be a nonempty floating (B,N,M) tensor")
        b, n, m = x.shape
        specs = (
            (self.categorical, x.shape, torch.int32),
            (self.categorical_mask, (b, m), torch.bool),
            (self.missing, x.shape, torch.bool),
            (self.row_uid, (b, n), torch.int64),
            (self.column_uid, (b, m), torch.int64),
        )
        for value, shape, dtype in specs:
            if value.shape != shape or value.dtype != dtype or value.device != x.device:
                raise ValueError(f"expected {shape}, {dtype}, device {x.device}")
        if self.n_columns is not None:
            d = self.n_columns
            if d.shape != (b,) or d.dtype != torch.int64 or d.device != x.device:
                raise ValueError(f"n_columns: expected {(b,)}, torch.int64, device {x.device}")
            if ((d < 1) | (d > m)).any():
                raise ValueError("n_columns must be in 1..M")
        for ids in (self.row_uid, self.column_uid):
            s = ids.sort(1).values
            if (s[:, 1:] == s[:, :-1]).any():
                raise ValueError("UIDs must be unique within each table")
        return self

    def rows(self, index):
        return replace(self, numerical=self.numerical[:, index],
                       categorical=self.categorical[:, index],
                       missing=self.missing[:, index], row_uid=self.row_uid[:, index])

    def columns(self, index):
        """Apply the same physical permutation to every batch member."""
        return replace(self, numerical=self.numerical[:, :, index],
                       categorical=self.categorical[:, :, index],
                       missing=self.missing[:, :, index],
                       categorical_mask=self.categorical_mask[:, index],
                       column_uid=self.column_uid[:, index])

    def canonical(self):
        self.validate()
        order = self.column_uid.argsort(1)
        idx = order[:, None].expand(-1, self.numerical.shape[1], -1)
        return Table(self.numerical.gather(2, idx), self.categorical.gather(2, idx),
                     self.categorical_mask.gather(1, order), self.missing.gather(2, idx),
                     self.row_uid, self.column_uid.gather(1, order), self.n_columns)


def folds_from_uid(uid):
    """Four folds from a deterministic integer hash; never use y or row order."""
    # Arithmetic stays in int64 for all signed UID values: reduce first.
    h = uid.remainder(2147483647)
    h = (h * 48271 + 12345).remainder(2147483647)
    h = torch.bitwise_xor(h, h >> 11)
    return h.remainder(4)


def canonical_labels(y, row_uid, n_classes):
    """Assign embedding slots by first stable observation, independent of class IDs.

    This makes simultaneous class relabeling exactly equivariant. An absent
    class has no retrieval mass and is assigned one of the remaining slots.
    """
    if y.dtype != torch.int64 or y.shape != row_uid.shape:
        raise ValueError("classification y must be int64 (B,N)")
    if not 2 <= n_classes <= 10 or ((y < 0) | (y >= n_classes)).any():
        raise ValueError("classification supports 2..10 classes, IDs 0..C-1")
    b, n = y.shape
    # Position of every row in UID order; first position of each class (n if absent).
    position = torch.empty_like(row_uid).scatter_(1, row_uid.argsort(1), torch.arange(n, device=y.device).expand(b, -1))
    first = torch.full((b, n_classes), n, device=y.device, dtype=torch.long)
    first.scatter_reduce_(1, y, position, reduce="amin", include_self=True)
    mapping = first.argsort(dim=-1, stable=True).argsort(-1)
    return mapping.gather(1, y), mapping
