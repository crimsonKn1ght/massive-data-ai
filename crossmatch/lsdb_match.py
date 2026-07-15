"""Real cross-match path: open two MMU HATS catalogs and stream matched image+spectrum records.

This uses the documented LSDB API - ``lsdb.open_catalog(path, columns=...)`` then
``left.crossmatch(right, n_neighbors=..., radius_arcsec=...)`` - and iterates the lazy result one
partition at a time so only overlapping sky tiles are materialized (the point of the HATS release).

Three things must be verified against the actual catalogs before a real run (they are config-driven,
never guessed here):

1. the exact HATS paths for the image and spectrum catalogs (``*_catalog.hats_path`` in the config);
2. that the matched frame carries the pixel/flux arrays (not coordinates only) - if a catalog is
   coordinates-only, the arrays must be joined from the base ``MultimodalUniverse/*`` dataset by id;
3. the exact column names for the image array, spectrum flux/wavelength, and redshift.

The loader logs the available columns on open, and raises a clear error listing them if a configured
column is absent, so a mismatch is obvious rather than silent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

import numpy as np

from .schema import AlignedRecord

logger = logging.getLogger(__name__)

_SPEC_SUFFIX = "_spec"  # applied to the spectrum (right) catalog columns during crossmatch


def _to_image_array(value: Any) -> np.ndarray:
    """Coerce a stored image cell to a ``(C, H, W)`` float32 array."""
    if isinstance(value, dict) and "array" in value:
        value = value["array"]
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]  # single band -> (1, H, W)
    if array.ndim != 3:
        raise ValueError(f"Expected an image array of ndim 2 or 3, got shape {array.shape}")
    return array


def _to_1d(value: Any) -> np.ndarray:
    """Coerce a stored spectrum cell (flux or wavelength) to a 1-D float32 array."""
    if isinstance(value, dict) and "array" in value:
        value = value["array"]
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array


def _require(row: Dict[str, Any], column: str, available: List[str]) -> Any:
    if column not in row:
        raise KeyError(
            f"Column {column!r} not found in the matched frame. Available columns: {available}. "
            "Update the *_column entries in the crossmatch config to match the opened catalog."
        )
    return row[column]


def crossmatched_records(config: Dict[str, Any], max_objects: int | None = None) -> Iterator[AlignedRecord]:
    """Yield ``AlignedRecord`` objects from the cross-match described by ``config``.

    ``config`` is the parsed ``configs/crossmatch_*.yaml`` (image_catalog / spectrum_catalog / match).
    """
    import lsdb  # imported lazily so the synthetic smoke path needs no astro stack

    img_cfg = config["image_catalog"]
    spec_cfg = config["spectrum_catalog"]
    match_cfg = config["match"]

    img_columns = [img_cfg["ra_column"], img_cfg["dec_column"], img_cfg["image_column"], *img_cfg.get("extra_columns", [])]
    spec_columns = [
        spec_cfg["ra_column"],
        spec_cfg["dec_column"],
        spec_cfg["flux_column"],
        spec_cfg["wavelength_column"],
        spec_cfg["redshift_column"],
        *spec_cfg.get("extra_columns", []),
    ]

    logger.info("Opening image catalog: %s (columns=%s)", img_cfg["hats_path"], img_columns)
    image_cat = lsdb.open_catalog(img_cfg["hats_path"], columns=img_columns)
    logger.info("Opening spectrum catalog: %s (columns=%s)", spec_cfg["hats_path"], spec_columns)
    spectrum_cat = lsdb.open_catalog(spec_cfg["hats_path"], columns=spec_columns)

    matched = image_cat.crossmatch(
        spectrum_cat,
        n_neighbors=int(match_cfg.get("n_neighbors", 1)),
        radius_arcsec=float(match_cfg.get("radius_arcsec", 1.0)),
        suffixes=("", _SPEC_SUFFIX),
    )

    image_col = img_cfg["image_column"]
    flux_col = spec_cfg["flux_column"] + _SPEC_SUFFIX
    wavelength_col = spec_cfg["wavelength_column"] + _SPEC_SUFFIX
    redshift_col = spec_cfg["redshift_column"] + _SPEC_SUFFIX
    ra_col = img_cfg["ra_column"]
    dec_col = img_cfg["dec_column"]

    n_yielded = 0
    # Iterate partition by partition (one overlapping sky tile at a time) so memory stays bounded.
    for delayed_partition in matched.to_delayed():
        partition = delayed_partition.compute()
        if partition is None or len(partition) == 0:
            continue
        available = list(partition.columns)
        for position, (_, row) in enumerate(partition.iterrows()):
            row = row.to_dict()
            image = _to_image_array(_require(row, image_col, available))
            flux = _to_1d(_require(row, flux_col, available))
            wavelength = _to_1d(_require(row, wavelength_col, available))
            redshift = _require(row, redshift_col, available)
            yield AlignedRecord(
                object_id=str(row.get("object_id", f"{n_yielded:09d}")),
                ra=float(row.get(ra_col, np.nan)),
                dec=float(row.get(dec_col, np.nan)),
                image=image,
                spectrum_flux=flux,
                spectrum_wavelength=wavelength,
                catalog={"redshift": float(redshift)},
            )
            n_yielded += 1
            if max_objects is not None and n_yielded >= max_objects:
                logger.info("Reached max_objects=%d; stopping.", max_objects)
                return
