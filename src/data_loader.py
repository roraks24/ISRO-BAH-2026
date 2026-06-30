"""
data_loader.py — Data acquisition and preprocessing for TESS light curves.

Supports:
  - Single-target download via lightkurve
  - Bulk sector download via lightkurve sector search
  - Curated dataset ingestion (CSV with TIC IDs + labels)
  - TOI catalog download from NASA Exoplanet Archive
"""

import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
from astroquery.mast import Catalogs


# ─────────────────────── single-target download ───────────────────────

def download_light_curve(target_name, mission="TESS", author="SPOC",
                         max_sectors=None, max_retries=3):
    """
    Download and stitch all available TESS sectors for a single target.
    Retries up to max_retries times on transient network errors.

    Returns
    -------
    lc : LightCurve
        Stitched, normalised light curve.
    tic_id : str
        Resolved TIC ID string.
    """
    import time as _time

    last_err = None
    for attempt in range(max_retries):
        try:
            search = lk.search_lightcurve(
                target_name, mission=mission, author=author
            )
            if len(search) == 0:
                # Try without author restriction as fallback
                search = lk.search_lightcurve(
                    target_name, mission=mission
                )
                if len(search) > 0:
                    available_authors = set(search.table["author"])
                    selected_author = None
                    for priority_author in ["SPOC", "TESS-SPOC", "QLP", "GSFC-ELEANOR-LITE"]:
                        if priority_author in available_authors:
                            selected_author = priority_author
                            break
                    if selected_author is None:
                        # Fallback to the most common author
                        from collections import Counter
                        counts = Counter(search.table["author"])
                        selected_author = counts.most_common(1)[0][0]
                    
                    print(f"  No {author} data. Fallback to author: {selected_author}")
                    search = search[search.table["author"] == selected_author]
            if len(search) == 0:
                raise ValueError(
                    f"No light curves found for {target_name}."
                )

            tic_id = str(search.table["target_name"][0])

            if max_sectors is not None:
                search = search[:max_sectors]

            lc_collection = search.download_all()
            lc = lc_collection.stitch()
            return lc, tic_id

        except ValueError:
            raise  # Don't retry on "not found" errors
        except Exception as e:
            last_err = e
            err_msg = str(e)
            
            # Check for corrupt files in lightkurve cache
            if "corrupt" in err_msg.lower() or "error in reading data product" in err_msg.lower():
                import re
                # Search for file path ending with .fits
                match = re.search(r'([A-Za-z]:\\[^\s]+\.fits|[^\s\']+\.fits)', err_msg)
                if match:
                    corrupt_file = match.group(1).strip('\'" ')
                    if os.path.exists(corrupt_file):
                        try:
                            print(f"  [Attempt {attempt+1}] Detected corrupt cache file: {corrupt_file}")
                            print("  Deleting corrupt file to force re-download...")
                            os.remove(corrupt_file)
                        except Exception as del_err:
                            print(f"  Failed to delete corrupt file {corrupt_file}: {del_err}")
            
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"  [Attempt {attempt+1}/{max_retries}] Network or read error, "
                  f"retrying in {wait}s: {type(e).__name__} ({e})")
            _time.sleep(wait)

    raise ConnectionError(
        f"Failed to download {target_name} after {max_retries} attempts: {last_err}"
    )


# ─────────────────────── bulk sector download ─────────────────────────

def search_sector_targets(sector, author="SPOC"):
    """
    Return a lightkurve SearchResult for every target observed in a sector.
    """
    search = lk.search_lightcurve(
        f"sector {sector}", mission="TESS", author=author, sector=sector
    )
    return search


def download_sector_lightcurves(sector, max_targets=None, cache_dir="data/sector_cache",
                                 author="SPOC"):
    """
    Download light curves for an entire TESS sector.

    Parameters
    ----------
    sector : int
    max_targets : int or None
        Cap for quick testing.
    cache_dir : str
        Directory to cache downloaded .npy arrays.

    Yields
    ------
    (tic_id, time, flux, flux_err) tuples for each successfully downloaded target.
    """
    os.makedirs(cache_dir, exist_ok=True)

    search = lk.search_lightcurve(
        "TESS", mission="TESS", author=author, sector=sector
    )
    if len(search) == 0:
        print(f"No results for sector {sector}. Trying target-list approach...")
        # Fallback: query MAST for the sector's target list
        from astroquery.mast import Observations
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries",
        )
        if len(obs) == 0:
            raise ValueError(f"No light curves found for sector {sector}.")
        tic_ids = list(set(obs["target_name"]))
        if max_targets:
            tic_ids = tic_ids[:max_targets]

        for tic_id in tqdm(tic_ids, desc=f"Sector {sector}"):
            try:
                lc, tid = download_light_curve(
                    f"TIC {tic_id}", max_sectors=1
                )
                t, f, fe = _extract_arrays(lc)
                if t is not None:
                    yield str(tic_id), t, f, fe
            except Exception as e:
                continue
        return

    # Normal path: lightkurve found the sector directly
    unique_targets = list(set(search.table["target_name"]))
    if max_targets:
        unique_targets = unique_targets[:max_targets]

    for target_name in tqdm(unique_targets, desc=f"Sector {sector}"):
        cache_file = os.path.join(cache_dir, f"{target_name}.npz")
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            yield str(target_name), data["time"], data["flux"], data["flux_err"]
            continue
        try:
            sub = search[search.table["target_name"] == target_name]
            lc = sub[0].download()
            if lc is None:
                continue
            t, f, fe = _extract_arrays(lc)
            if t is None:
                continue
            np.savez_compressed(cache_file, time=t, flux=f, flux_err=fe)
            yield str(target_name), t, f, fe
        except Exception:
            continue


def _extract_arrays(lc):
    """Clean a lightkurve LightCurve and return (time, flux, flux_err) arrays."""
    try:
        lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
        t = np.asarray(lc.time.value, dtype=np.float64)
        f = np.asarray(lc.flux.value, dtype=np.float64)
        fe = np.asarray(lc.flux_err.value, dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(fe)
        t, f, fe = t[mask], f[mask], fe[mask]
        if len(t) < 100:
            return None, None, None
        return t, f, fe
    except Exception:
        return None, None, None


# ─────────────────── curated dataset ingestion ────────────────────────

def load_curated_dataset(csv_path):
    """
    Load a curated labeled dataset.

    Expected CSV columns:
        tic_id, label, [optional: period, depth, duration]
    Labels should be one of:
        planet, eclipsing_binary, blend, starspot, false_alarm
    """
    df = pd.read_csv(csv_path)
    required = {"tic_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required}, got {set(df.columns)}"
        )
    # Normalise labels
    label_map = {
        "cp": "planet", "kp": "planet", "pc": "planet",
        "planet": "planet", "planet_candidate": "planet",
        "fp": "false_positive", "false_positive": "false_positive",
        "eb": "eclipsing_binary", "eclipsing_binary": "eclipsing_binary",
        "blend": "blend", "blend_or_false_positive": "blend",
        "starspot": "starspot", "starspot_activity": "starspot",
        "fa": "false_alarm", "false_alarm": "false_alarm",
    }
    df["label"] = df["label"].str.strip().str.lower().map(label_map).fillna("other")
    return df


# ─────────────────── TOI catalog download ─────────────────────────────

def load_toi_catalog():
    """
    Download the TESS Objects of Interest (TOI) catalog from
    the NASA Exoplanet Archive via TAP.

    Returns a DataFrame with columns including:
        tid (TIC ID), toipfx, tfopwg_disp (label),
        pl_orbper, pl_trandep, pl_trandur, st_rad, st_teff
    """
    import requests
    from io import StringIO

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = """
    SELECT toipfx, tid, tfopwg_disp,
           pl_orbper, pl_trandep, pl_trandur,
           pl_rade, st_rad, st_teff, st_tmag
    FROM toi
    WHERE tfopwg_disp IS NOT NULL
    """
    params = {"query": query, "format": "csv"}
    print("Downloading TOI catalog from NASA Exoplanet Archive...")
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    print(f"Downloaded {len(df)} TOI entries")
    print(df["tfopwg_disp"].value_counts().to_string())

    # Map labels to our standard classes
    label_map = {
        "CP": "planet", "KP": "planet",
        "FP": "false_positive",
        "FA": "false_alarm",
        "PC": "candidate",
    }
    df["label"] = df["tfopwg_disp"].map(label_map).fillna("candidate")
    df["tic_id"] = df["tid"].astype(str)
    return df


# ─────────────────── stellar parameters ───────────────────────────────

def get_stellar_params(tic_id):
    """
    Query the TIC catalog for stellar radius, Teff, and magnitude.
    Returns dict with keys: radius_rsun, teff_k, tmag.
    """
    import re
    match = re.search(r"\d+", str(tic_id))
    if not match:
        return {"radius_rsun": 1.0, "teff_k": 5778.0, "tmag": 10.0}

    numeric_id = match.group()
    try:
        result = Catalogs.query_criteria(catalog="Tic", ID=numeric_id)
        rad = float(result["rad"][0])
        teff = float(result["Teff"][0])
        tmag = float(result["Tmag"][0])
        if not np.isfinite(rad):
            rad = 1.0
        if not np.isfinite(teff):
            teff = 5778.0
        if not np.isfinite(tmag):
            tmag = 10.0
        return {"radius_rsun": rad, "teff_k": teff, "tmag": tmag}
    except Exception:
        return {"radius_rsun": 1.0, "teff_k": 5778.0, "tmag": 10.0}
