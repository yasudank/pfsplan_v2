import csv
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, get_body, SkyCoord, Angle
from astropy.time import Time
from astropy.table import Table
from astroplan import FixedTarget
import warnings
from obs_utils import setup_observer, read_obsdates, read_priorities
import datetime
import yaml
import concurrent.futures

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_config(filename='obs_config.yaml'):
    """
    Load observation parameters from a YAML file.
    """
    try:
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {filename} not found. Using defaults.")
        # Return default configuration matching original hardcoded values
        return {
            'slew': {
                'speed_az': 0.5,
                'speed_el': 0.5,
                'speed_rot': 1.5
            },
            'constraints': {
                'max_airmass': 1.6,
                'min_altitude': 0.0,
                'min_teff': 0.6,
                'rotator_min': -174.0,
                'rotator_max': 174.0
            },
            'scheduling': {
                'min_overhead_min': 5,
                'manual_readout_min': 15,
                'overlap_separation_deg': 1.4,
                'adjacency_bonus_window_sec': 5.0,
                'adjacency_bonus_score': 1000.0,
                'slew_penalty_weight': 0.5
            }
        }

# --- Copied from Moon.py ---
class MoonBrightnessModel:
    def __init__(self):
        self.k = {
            "g": 0.15, "r": 0.10, "i": 0.09, "z": 0.08, "y": 0.1
        }
        self.Q = {
            "g": 0.129, "r": 0.152, "i": 0.114, "z": 0.048, "y": 0.038
        }
        self.mu_sky = {
            "g": 22.25, "r": 21.18, "i": 20.32, "z": 19.59, "y": 18.28
        }
        self.lam_eff = {
            "g": 478.0, "r": 617.0, "i": 766.0, "z": 888.0, "y": 974.0
        }
        self.Msun = {
            "g": -26.520, "r": -26.922, "i": -27.042, "z": -27.054, "y": -27.059, "V": -26.756
        }

    # Airmass for zenith distance z (deg)
    def X(self, z):
        z = np.radians(z)
        return 1.0 / np.sqrt(1.0 - 0.96 * np.sin(z)**2)

    # Rayleigh scattering
    def tR(self, band, X):
        p = 608.0  # Pressure at Mauna Kea (hPa)
        H = 4.2  # Height of Mauna Kea (km)
        lam = self.lam_eff[band] * 1.0E-03
        tauR = p / 1013.25 * (0.00864 + 6.5E-06 * H) * lam**(-(3.916 + 0.074 * lam + 0.050 / lam))
        return np.exp(-tauR * X)

    # Mie scattering
    def tM(self, band, X):
        lam = self.lam_eff[band] * 1.0E-03
        alpha = -1.38
        kM = np.where(lam < 0.4, 0.050, 0.013 * lam**alpha)
        return 10.0**(-0.4 * kM * X)

    def Bmoon(self, band, alpha, z_moon, X_sky, rho):
        # band: filter band
        # alpha: lunar phase angle (deg)
        # z_moon: lunar zenith distance (deg)
        # X_sky: airmass for the sky
        # rho: angular separation between the Moon and the field (deg)
        XV = self.Msun[band] - self.Msun["V"]
        phi = 180 - alpha
        Istar = 10.0**(-0.4 * (3.84 + 0.026 * np.abs(phi) + 4.0E-09 * phi**4)) * 10.0**(-0.4 * XV)
        X_moon = self.X(z_moon)
        rho = np.radians(rho)
        fR = 10.0**0.92 * (1.06 + np.cos(rho)**2)
        fM = 10.0**(2.44 - np.degrees(rho) / 40.0)
        BmoonR = fR * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tR(band, X_sky))
        BmoonM = fM * Istar * 10.0**(-0.4 * self.k[band] * X_moon) * (1.0 - self.tM(band, X_sky))
        return BmoonR + BmoonM

    def deltaMag(self, band, alpha, z_moon, z_sky, rho):
        if np.any(z_moon < 90.0):
            X_sky = self.X(z_sky)
            B0 = self.Q[band] * 5.48E+06 * 10.0**(-0.4 * self.mu_sky[band]) * X_sky
            Bm = self.Bmoon(band, alpha, z_moon, X_sky, rho)
            return -2.5 * np.log10((Bm + B0) / B0)
        else:
            return np.zeros_like(z_moon)

class SlewParams:
    def __init__(self, config):
        self.slew_speed_az = config['slew']['speed_az'] # degree / sec
        self.slew_speed_el = config['slew']['speed_el'] # degree / sec
        self.inst_rot_speed = config['slew']['speed_rot'] # degree / sec

def calculate_slew_time(cur_altaz, cur_rotang, tgt_altaz, tgt_rotang, params, verbose=False):
    """
    Calculate the slew time based on the current and target altaz and rotator angles.
    """
    rate_az = params.slew_speed_az
    rate_el = params.slew_speed_el
    rate_rot = params.inst_rot_speed

    # Calculate the difference of the azimuth
    # If the difference is larger than 180 degrees, the telescope should rotate in the opposite direction
    # .az and .alt are Angles/Quantities, so convert to deg float for division
    az_diff = (tgt_altaz.az - cur_altaz.az).wrap_at(180 * u.deg).to(u.deg).value
    el_diff = (tgt_altaz.alt - cur_altaz.alt).to(u.deg).value
    
    # Rotator angles are assumed to be floats (degrees) or Angles
    if isinstance(cur_rotang, u.Quantity) or isinstance(cur_rotang, Angle):
        cur_rot = cur_rotang.to(u.deg).value
    else:
        cur_rot = cur_rotang
        
    if isinstance(tgt_rotang, u.Quantity) or isinstance(tgt_rotang, Angle):
        tgt_rot = tgt_rotang.to(u.deg).value
    else:
        tgt_rot = tgt_rotang

    # Simple difference for rotator (assuming no wrap issues or handled externally, 
    # though rotator usually has limits -180 to 180. 
    # The instrument rotator might not wrap continuously like azimuth?)
    # Assuming simple difference for now as per original slew.py
    rot_diff = tgt_rot - cur_rot

    # Debugging output
    if verbose:
        print(f"[Verbose-Slew] cur_altaz: Alt={cur_altaz.alt.deg:.2f}, Az={cur_altaz.az.deg:.2f}, cur_rotang={cur_rot:.2f}")
        print(f"[Verbose-Slew] tgt_altaz: Alt={tgt_altaz.alt.deg:.2f}, Az={tgt_altaz.az.deg:.2f}, tgt_rotang={tgt_rot:.2f}")
        print(f"[Verbose-Slew] az_diff={az_diff:.2f}, el_diff={el_diff:.2f}, rot_diff={rot_diff:.2f}")

    # Calculate the slew time using a simple model
    slew_time = max(abs(az_diff) / rate_az,
                    abs(el_diff) / rate_el,
                    abs(rot_diff) / rate_rot)
    
    if verbose:
        print(f"[Verbose-Slew] Calculated slew_time (numeric): {slew_time:.2f}")

    return slew_time * u.s
# --- End of Moon.py copy ---

def calculate_teff(observer, target_coord, target_alt, target_airmass, moon_coord, moon_altaz, moon_phase, mbm):
    observer_lat = observer.location.lat.deg
    target_dec = target_coord.dec.deg
    
    zmin = abs(target_dec - observer_lat)
    if zmin > 89.9: zmin = 89.9
            
    airmass0 = mbm.X(zmin)
    if airmass0 > 100: airmass0 = 100 # cap airmass
    teff0 = 1.0 / (airmass0 * 10**(0.8*mbm.k['r']*(airmass0-1.0)))
    if teff0 == 0: return 0 # avoid division by zero

    airmass = target_airmass    
    z_obs = 90. - target_alt
    z_moon = 90. - moon_altaz.alt.deg
    
    if z_moon >= 90.:
        dmu = 0.0
    else:
        moon_sep = moon_coord.separation(target_coord).deg
        dmu = mbm.deltaMag("r",
                           moon_phase.to(u.deg).value,
                           z_moon,
                           z_obs,
                           moon_sep)

    teff_abs = (1.0 / (10**(-0.4*dmu) * airmass * 10**(0.8*mbm.k['r']*(airmass-1.0))))
    
    return teff_abs / teff0

def check_visibility_worker(args):
    """
    Worker function to check visibility for a single target.
    Arguments are packed in a tuple to be compatible with map/executor.
    """
    (t, current_time, current_pointing, cur_altaz, current_rotator_angle, 
     next_reservation_start, observer, slew_params, min_overhead, 
     max_airmass, rot_min, rot_max, min_teff, 
     moon_coord, moon_altaz, moon_phase, mbm, verbose) = args

    try:
        # Calculate potential overhead including slew
        tgt_altaz_for_slew = observer.altaz(current_time, t['target']) 
        
        # Calculate parallactic angle for current time
        pa_angle = observer.parallactic_angle(current_time, t['target']).to(u.deg).value
        raw_rotator_angle = pa_angle + t['ppc_pa']
        rotator_angle = Angle(raw_rotator_angle * u.deg).wrap_at(180 * u.deg).value
        
        if current_pointing is not None:
            slew_time_val = calculate_slew_time(cur_altaz, current_rotator_angle, tgt_altaz_for_slew, rotator_angle, slew_params, verbose=verbose)
            overhead = min_overhead + slew_time_val
        else:
            overhead = min_overhead
        
        # Check if fits in gap
        req_duration = t['exptime']*u.s + overhead
        if current_time + req_duration > next_reservation_start:
            return None
        
        # Calculate alt/airmass and rotator angle for the actual mid-point of exposure
        obs_start_time_for_eval = current_time + overhead
        obs_mid_time_for_eval = obs_start_time_for_eval + t['exptime']*u.s / 2
        
        altaz_obs_mid = observer.altaz(obs_mid_time_for_eval, t['target'])
        alt = altaz_obs_mid.alt.deg
        airmass = altaz_obs_mid.secz
        
        pa_angle_obs_mid = observer.parallactic_angle(obs_mid_time_for_eval, t['target']).to(u.deg).value
        rotator_angle_obs_mid = Angle((pa_angle_obs_mid + t['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value

        if airmass <= max_airmass and alt > 0:
            # Check instrument rotator angle constraint at observation mid-point
            if rotator_angle_obs_mid < rot_min or rotator_angle_obs_mid > rot_max:
                return None
                
            teff = calculate_teff(observer, t['target'].coord, alt, airmass, moon_coord, moon_altaz, moon_phase, mbm)
            if teff > min_teff:
                return {
                    'info': t,
                    'alt': alt,
                    'airmass': airmass,
                    'coord': t['target'].coord,
                    'teff': teff,
                    'rotator_angle': rotator_angle_obs_mid,
                    'overhead': overhead
                }
    except Exception as e:
        # In case of calculation error, just skip
        if verbose:
            print(f"Error checking target {t['id']}: {e}")
        return None
        
    return None

def load_all_targets(priorities):
    """
    Load targets from CO, GA, and GE summary files.
    Returns a dictionary of target info keyed by ppc_code.
    """
    all_targets = {}
    files = ['CO_summary_reconfigure.csv', 'GA_summary_reconfigure.csv', 'GE_summary_reconfigure.csv']
    
    for filename in files:
        try:
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                ppc_code = row['ppc_code']
                # Only CO targets usually have priorities. Others default to 99.
                priority = priorities.get(ppc_code, 99)
                
                coord = SkyCoord(ra=row['ppc_ra']*u.deg, dec=row['ppc_dec']*u.deg)
                target = FixedTarget(coord=coord, name=ppc_code)
                
                all_targets[ppc_code] = {
                    'id': ppc_code,
                    'target': target,
                    'exptime': float(row['ppc_exptime']),
                    'observed': False,
                    'priority': priority,
                    'nframes': int(row['ppc_nframes']) if 'ppc_nframes' in row else 1,
                    'ppc_pa': float(row['ppc_pa'])
                }
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
            
    return all_targets

def load_all_targets_from_ppcList(priorities):
    """
    Load targets from CO, GA, and GE ppcList files.
    Returns a dictionary of target info keyed by ppc_code.
    """
    all_targets = {}
    files = ['targets/CO/ppcList.ecsv', 'targets/GA/ppcList.ecsv', 'targets/GE/ppcList.ecsv']
    
    for filename in files:
        try:
            df = Table.read(filename, format='ascii.ecsv').to_pandas()
            for _, row in df.iterrows():
                ppc_code = row['ppc_code']
                # Only CO targets usually have priorities. Others default to 99.
                priority = priorities.get(ppc_code, 99)
                
                coord = SkyCoord(ra=row['ppc_ra']*u.deg, dec=row['ppc_dec']*u.deg)
                target = FixedTarget(coord=coord, name=ppc_code)
                
                all_targets[ppc_code] = {
                    'id': ppc_code,
                    'target': target,
                    'exptime': float(row['ppc_exptime']),
                    'observed': False,
                    'priority': priority,
                    'nframes': int(row['ppc_nframes']) if 'ppc_nframes' in row else 1,
                    'ppc_pa': float(row['ppc_pa'])
                }
        except FileNotFoundError:
            print(f"Warning: {filename} not found.")
            
    return all_targets

def load_manual_schedule(filename):
    """
    Load manual allocation schedule.
    Returns a dictionary mapping date string (HST) to list of targets.
    """
    manual_schedule = {} 
    try:
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            date = row['obs_date']
            if date not in manual_schedule:
                manual_schedule[date] = []
            manual_schedule[date].append({
                'ppc_code': row['ppc_code'],
                'nframes': int(row['ppc_nframes'])
            })
    except Exception as e:
        print(f"Error reading manual allocation: {e}")
    return manual_schedule

def find_optimal_slot(observer, target_info, duration, night_start, night_end, busy_slots, time_step, target_ppc_pa, config):
    """
    Find the best time slot for a target within the night to maximize altitude.
    Must not overlap with busy_slots.
    Prioritizes slots adjacent to existing busy_slots to minimize gaps by adding specific candidates
    and applying a large score bonus.
    Also, filters out slots where the instrument rotator angle is out of bounds.
    """
    best_time = None
    best_score = -np.inf
    
    # Config parameters
    rot_min = config['constraints']['rotator_min']
    rot_max = config['constraints']['rotator_max']
    max_airmass = config['constraints']['max_airmass']
    adj_window = config['scheduling']['adjacency_bonus_window_sec']
    adj_bonus = config['scheduling']['adjacency_bonus_score']
    
    # Debug statistics
    fail_stats = {
        'overlap': 0,
        'rotator': 0,
        'airmass': 0,
        'altitude': 0,
        'total_checked': 0
    }

    # 1. Generate candidate times (base grid)
    candidates = []
    curr = night_start
    while curr + duration <= night_end:
        candidates.append(curr)
        curr += time_step
        
    # 2. Add adjacent candidates
    for b_start, b_end in busy_slots:
        # Immediately after an existing slot
        if b_end + duration <= night_end:
            candidates.append(b_end)
        # Immediately before an existing slot
        if b_start - duration >= night_start:
            candidates.append(b_start - duration)
            
    # Filter valid times
    valid_times = []
    for t_start in candidates:
        t_end = t_start + duration
        
        # Bounds check
        if t_start < night_start or t_end > night_end:
            continue
            
        # Overlap check
        is_valid = True
        for b_start, b_end in busy_slots:
            # Intersection: not (End <= Start OR Start >= End)
            # Use a small tolerance (e.g. 1 sec) to allow touching
            # Intersection occurs if (my_start < b_end - tol) and (my_end > b_start + tol)
            if (t_start < b_end - 1*u.s) and (t_end > b_start + 1*u.s):
                is_valid = False
                break
        
        fail_stats['total_checked'] += 1
        if is_valid:
            valid_times.append(t_start)
        else:
            fail_stats['overlap'] += 1

    if not valid_times:
        if fail_stats['total_checked'] > 0:
            print(f"    - [Debug] {target_info['id']}: No valid slots found (checked {fail_stats['total_checked']}). All overlapped or out of bounds.")
        return None

    # Evaluate altitudes for valid slots
    for t_start in valid_times:
        t_end = t_start + duration
        t_mid = t_start + duration / 2
        
        # Calculate parallactic angle for current time
        # Note: parallactic_angle returns a Quantity, convert to value for comparison
        pa_angle = observer.parallactic_angle(t_mid, target_info['target']).to(u.deg).value
        
        # Calculate rotator angle and wrap at 180 degrees
        raw_rotator_angle = pa_angle + target_ppc_pa
        rotator_angle = Angle(raw_rotator_angle * u.deg).wrap_at(180 * u.deg).value

        # Check instrument rotator angle constraint
        if rotator_angle < rot_min or rotator_angle > rot_max:
            fail_stats['rotator'] += 1
            continue # Skip this slot as it's outside the allowed rotator angle range
            
        # Calculate altitude/airmass at mid-point
        altaz = observer.altaz(t_mid, target_info['target'])
        alt = altaz.alt.deg
        airmass = altaz.secz
        
        if airmass > max_airmass:
            fail_stats['airmass'] += 1
            continue
        
        if alt <= 0: # Check altitude > 0
            fail_stats['altitude'] += 1
            continue
            
        # Adjacency Bonus
        bonus = 0
        for b_start, b_end in busy_slots:
            if abs((t_start - b_end).sec) < adj_window or abs((t_end - b_start).sec) < adj_window:
                bonus = adj_bonus # Massive bonus to force adjacency
                break
        
        score = alt + bonus
        
        # Only consider valid altitudes (and prefer higher score)
        if score > best_score:
            best_score = score
            best_time = t_start
            
    if best_time is None:
        print(f"    - [Debug] {target_info['id']}: Failed to find slot. Reasons: Overlap={fail_stats['overlap']}, Rotator={fail_stats['rotator']}, Airmass={fail_stats['airmass']}, Altitude={fail_stats['altitude']} (from {fail_stats['total_checked']} candidates).")

    return best_time

def run_scheduler(observer, all_targets, manual_schedule, nights, config, verbose=False):
    """
    Run the scheduler with manual allocation and greedy auto-scheduling.
    """
    schedule = []
    mbm = MoonBrightnessModel()
    slew_params = SlewParams(config)

    # Standard overhead (minimum) and manual observation block settings
    min_overhead = config['scheduling']['min_overhead_min'] * u.min
    manual_readout_exptime = config['scheduling']['manual_readout_min'] * u.min
    manual_block_len = min_overhead + manual_readout_exptime # 20 min per frame
    
    # Constraints
    max_airmass = config['constraints']['max_airmass']
    min_teff = config['constraints']['min_teff']
    rot_min = config['constraints']['rotator_min']
    rot_max = config['constraints']['rotator_max']
    overlap_sep = config['scheduling']['overlap_separation_deg']
    slew_penalty = config['scheduling']['slew_penalty_weight']
    
    observed_history = []

    # Get unique priorities for auto-scheduling (exclude default 99)
    greedy_priorities = sorted(list(set(t['priority'] for t in all_targets.values() if t['priority'] < 99)))
    
    executor = concurrent.futures.ProcessPoolExecutor()
    try:
        for night_idx, (start_time, end_time) in enumerate(nights):
            current_time = start_time
            current_pointing = None
            current_rotator_angle = 0.0 # Initialize rotator angle
            
            print(f"\n=== Night {night_idx+1}: {start_time.iso} to {end_time.iso} ===")
            
            # --- 1. Reserve Manual Targets ---
            # Get HST date string for manual schedule lookup
            hst_date_str = (start_time - 10*u.hour).to_datetime().date().isoformat()
            reservations = [] # List of (start, end, target_info, nframes)
            
            if hst_date_str in manual_schedule:
                print(f"  [Manual] Scheduling fixed targets for {hst_date_str}...")
                manual_requests = manual_schedule[hst_date_str]
                
                # Sort requests
                # Use requests in the order they appear in the CSV to preserve user intent
                sorted_requests = manual_requests
                
                # Helper to check constraints for a specific time slot
                def check_slot_constraints(t_start, t_end, t_info):
                    t_mid = t_start + (t_end - t_start)/2
                    # Rotator
                    pa = observer.parallactic_angle(t_mid, t_info['target']).to(u.deg).value
                    rot = Angle((pa + t_info['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value
                    if not (config['constraints']['rotator_min'] <= rot <= config['constraints']['rotator_max']):
                        return False
                    # Airmass
                    altaz = observer.altaz(t_mid, t_info['target'])
                    if altaz.alt.deg <= 0: return False
                    if altaz.secz > config['constraints']['max_airmass']:
                         return False
                    return True
                
                # Group requests by target proximity to optimize contiguous blocks
                grouped_requests = []
                if sorted_requests:
                    current_group = [sorted_requests[0]]
                    first_info = all_targets[sorted_requests[0]['ppc_code']]
                    current_coord = first_info['target'].coord
                    
                    for i in range(1, len(sorted_requests)):
                        req = sorted_requests[i]
                        info = all_targets[req['ppc_code']]
                        coord = info['target'].coord
                        
                        if current_coord.separation(coord) < 5.0 * u.deg:
                            current_group.append(req)
                        else:
                            grouped_requests.append(current_group)
                            current_group = [req]
                            current_coord = coord
                    grouped_requests.append(current_group)
                
                # Loop over groups
                for group_idx, group in enumerate(grouped_requests):
                    # Representative info (first in group)
                    rep_req = group[0]
                    rep_ppc_code = rep_req['ppc_code']
                    rep_info = all_targets[rep_ppc_code]
                    
                    # Calculate total duration
                    group_nframes = sum(r['nframes'] for r in group)
                    total_duration = group_nframes * manual_block_len
                    
                    # Determine search start (Constraint Logic)
                    search_start = start_time
                    
                    # Find slot for combined block
                    busy = [(r[0], r[1]) for r in reservations]
                    
                    # Using representative target for optimization
                    start_slot = find_optimal_slot(observer, rep_info, total_duration, search_start, end_time, busy, 1*u.min, rep_info['ppc_pa'], config)
                    
                    if start_slot:
                        # --- Gap Check Logic ---
                        # Use variable names consistent with logic below
                        ppc_code = rep_ppc_code + " (Group)"
                        target_info = rep_info # For constraints check
                        
                        # Define minimum gap for Auto observations (4 frames)
                        min_auto_gap = 4 * (min_overhead + 15 * u.min) # ~80 min
                        
                        # Estimate effective Auto block length (including avg slew)
                        est_auto_len = min_overhead + 15 * u.min + 1 * u.min
                        
                        # Calculate potential end slot
                        end_slot = start_slot + total_duration
                        
                        # Find the nearest busy block before and after
                        nearest_end_before = start_time
                        nearest_start_after = end_time
                        
                        for b_start, b_end in busy:
                            # Add tolerance for float comparison
                            if b_end <= start_slot + 1*u.s and b_end > nearest_end_before:
                                nearest_end_before = b_end
                            if b_start >= end_slot - 1*u.s and b_start < nearest_start_after:
                                nearest_start_after = b_start
                                
                        gap_before = start_slot - nearest_end_before
                        gap_after = nearest_start_after - end_slot
                        
                        shifted = False
                        
                        # 1. Close small gaps (< 80 min) completely
                        if 0 < gap_before.to(u.min).value < min_auto_gap.to(u.min).value:
                            if check_slot_constraints(nearest_end_before, nearest_end_before + total_duration, target_info):
                                print(f"    - [Adjust] Closing small gap before {ppc_code} ({gap_before.to(u.min):.1f}). Shifting to {nearest_end_before.iso}")
                                start_slot = nearest_end_before
                                end_slot = start_slot + total_duration
                                shifted = True
                        
                        # 2. Optimize large gaps (> 80 min) to avoid wasted tail
                        elif gap_before.to(u.min).value >= min_auto_gap.to(u.min).value:
                            remainder = (gap_before.to(u.min).value) % est_auto_len.to(u.min).value
                            if 0 < remainder < (min_overhead + 15*u.min).to(u.min).value:
                                 shift_amount = remainder * u.min
                                 new_start = start_slot - shift_amount
                                 
                                 # Constraint check
                                 valid_shift = True
                                 if not check_slot_constraints(new_start, new_start + total_duration, target_info):
                                     valid_shift = False
                                         
                                 if valid_shift:
                                     print(f"    - [Adjust] Optimizing large gap before {ppc_code}. Remainder {remainder:.1f}m. Shifting earlier by {shift_amount:.1f}")
                                     start_slot = new_start
                                     end_slot = start_slot + total_duration
                                     shifted = True

                        # Then check 'after' gap (using new end_slot)
                        if not shifted:
                            gap_after = nearest_start_after - end_slot
                            
                            should_check_gap_after = True
                            # Avoid shifting to end of night if there are more manual groups to come
                            # This prevents reversing the order of sequential groups
                            if abs((nearest_start_after - end_time).to(u.s).value) < 1.0 and group_idx < len(grouped_requests) - 1:
                                should_check_gap_after = False

                            if should_check_gap_after and 0 < gap_after.to(u.min).value < min_auto_gap.to(u.min).value:
                                 # Check if shifting later fits
                                 potential_start = nearest_start_after - total_duration
                                 if potential_start >= nearest_end_before: 
                                     if check_slot_constraints(potential_start, nearest_start_after, target_info):
                                         print(f"    - [Adjust] Closing small gap after {ppc_code} ({gap_after.to(u.min):.1f}). Shifting to {potential_start.iso}")
                                         start_slot = potential_start
                                         end_slot = nearest_start_after
                                         shifted = True

                        # Decompose group and schedule individual targets
                        curr_slot_start = start_slot
                        for req in group:
                            req_code = req['ppc_code']
                            req_info = all_targets[req_code]
                            req_frames = req['nframes']
                            req_dur = req_frames * manual_block_len
                            
                            req_end = curr_slot_start + req_dur
                            reservations.append((curr_slot_start, req_end, req_info, req_frames))
                            
                            mid_alt = observer.altaz(curr_slot_start + req_dur/2, req_info['target']).alt.deg
                            print(f"    - Scheduled {req_code} ({req_frames} frames) at {curr_slot_start.iso} (Avg Alt: {mid_alt:.1f})")
                            req_info['observed'] = True 
                            
                            curr_slot_start = req_end
                    else:
                        print(f"    - [Warning] Could not find slot for group starting with {rep_ppc_code}!")

            # Sort reservations by start time
            reservations.sort(key=lambda x: x[0])
            
            # --- 1b. Compact Manual Schedule (Post-processing) ---
            # Close gaps < min_auto_gap between manual blocks to prevent single Auto frames
            # min_auto_gap is ~80 min
            min_auto_gap = 4 * (min_overhead + 15 * u.min)
            
            # 1. Check Gap from Night Start to First Block
            if reservations:
                r_start, r_end, r_target, r_nframes = reservations[0]
                gap = r_start - start_time
                if 0 < gap.to(u.min).value < min_auto_gap.to(u.min).value:
                    # Try to shift earlier to start_time
                    new_start = start_time
                    new_end = new_start + (r_end - r_start)
                    
                    if check_slot_constraints(new_start, new_end, r_target):
                        print(f"    - [Compact] Closing start gap ({gap.to(u.min):.1f}). Shifting {r_target['id']} to {new_start.iso}")
                        reservations[0] = (new_start, new_end, r_target, r_nframes)

            # 2. Check Gaps between Blocks
            for i in range(len(reservations) - 1):
                curr_start, curr_end, curr_target, curr_nframes = reservations[i]
                next_start, next_end, next_target, next_nframes = reservations[i+1]
                
                gap = next_start - curr_end
                
                if 0 < gap.to(u.min).value < min_auto_gap.to(u.min).value:
                    # Try to shift 'next' earlier to 'curr_end'
                    duration = next_end - next_start
                    new_start = curr_end
                    new_end = new_start + duration
                    
                    if check_slot_constraints(new_start, new_end, next_target):
                        print(f"    - [Compact] Closing inter-block gap ({gap.to(u.min):.1f}). Shifting {next_target['id']} to {new_start.iso}")
                        reservations[i+1] = (new_start, new_end, next_target, next_nframes)
                        # Update local variable for next iteration? No, next iteration uses i+1 as curr.
                        # But we updated reservations list, so next iter will see new values. Correct.

            # 3. Check Gap from Last Block to Night End
            if reservations:
                last_r_start, last_r_end, _, _ = reservations[-1]
                gap = end_time - last_r_end
                
                if 0 < gap.to(u.min).value < min_auto_gap.to(u.min).value:
                    # Identify contiguous block from the end
                    indices_to_shift = [len(reservations) - 1]
                    for i in range(len(reservations) - 2, -1, -1):
                        curr_end = reservations[i][1]
                        next_start = reservations[i+1][0]
                        # If gap between blocks is negligible (was closed or already small)
                        # Step 2 closes gaps < min_auto_gap, so connected blocks should have 0 gap.
                        if (next_start - curr_end).to(u.min).value < 1.0: # 1 min tolerance
                             indices_to_shift.insert(0, i)
                        else:
                             break
                    
                    # Check constraints for all blocks in cluster
                    valid_shift = True
                    shift_amount = gap # Shift exactly to end
                    
                    # Check if shifting all these targets is valid
                    for idx in indices_to_shift:
                        r_start, r_end, r_target, r_nframes = reservations[idx]
                        new_start = r_start + shift_amount
                        new_end = r_end + shift_amount
                        if not check_slot_constraints(new_start, new_end, r_target):
                            valid_shift = False
                            print(f"    - [Compact] Cannot shift block ending with {reservations[-1][2]['id']} because {r_target['id']} would violate constraints.")
                            break
                    
                    if valid_shift:
                         print(f"    - [Compact] Closing end gap ({gap.to(u.min):.1f}). Shifting {len(indices_to_shift)} blocks to end of night.")
                         for idx in indices_to_shift:
                             r_start, r_end, r_target, r_nframes = reservations[idx]
                             new_start = r_start + shift_amount
                             new_end = r_end + shift_amount
                             reservations[idx] = (new_start, new_end, r_target, r_nframes)

            # Mark reserved manual targets as observed
            for r_start, r_end, r_target, r_nframes in reservations:
                r_target['observed'] = True # Mark as observed for greedy algorithm

            # --- 2. Fill Gaps with Greedy Algorithm ---
            while current_time < end_time:
                
                # Check for upcoming reservation
                active_reservation = None
                next_reservation_start = end_time
                
                for r_start, r_end, r_target, r_nframes in reservations:
                    if r_start <= current_time < r_end:
                        active_reservation = (r_start, r_end, r_target, r_nframes)
                        break
                    if r_start > current_time:
                        next_reservation_start = r_start
                        break
                
                # Case 1: Current time is within an active manual reservation
                if active_reservation:
                    r_start, r_end, r_target, r_nframes = active_reservation
                    
                    # Align current_time if we drifted slightly or skipped
                    # We log individual frames
                    base_time = max(current_time, r_start)
                    
                    for i in range(r_nframes):
                        f_start = r_start + i * manual_block_len
                        f_end = f_start + manual_block_len
                        
                        # Don't log if it's already past (shouldn't happen with proper logic)
                        if f_end <= current_time: 
                            continue

                        mid_time = f_start + manual_block_len/2
                        altaz = observer.altaz(mid_time, r_target['target'])
                        
                        # Calculate Rotator Angle (Start/End)
                        pa_start = observer.parallactic_angle(f_start, r_target['target']).to(u.deg).value
                        rot_start = Angle((pa_start + r_target['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value
                        
                        pa_end = observer.parallactic_angle(f_end, r_target['target']).to(u.deg).value
                        rot_end = Angle((pa_end + r_target['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value

                        # Calculate LST
                        lst = observer.local_sidereal_time(f_start).to_string(sep=':', precision=0)
                        
                        # Moon Stats
                        moon_coord = get_body('moon', f_start, location=observer.location)
                        #moon_sep = r_target['target'].coord.separation(moon_coord).deg
                        moon_sep = moon_coord.separation(r_target['target'].coord).deg

                        moon_altaz_obj = observer.altaz(f_start, moon_coord)
                        moon_alt = moon_altaz_obj.alt.deg
                        moon_illum = observer.moon_illumination(f_start)
                        
                        # Teff
                        sun = get_body('sun', f_start, location=observer.location)
                        moon_phase = moon_coord.separation(sun, origin_mismatch="ignore")
                        teff = calculate_teff(observer, r_target['target'].coord, altaz.alt.deg, altaz.secz, moon_coord, moon_altaz_obj, moon_phase, mbm)

                        schedule.append({
                            'night': night_idx + 1,
                            'target': r_target['id'],
                            'start_time': f_start.iso,
                            'end_time': f_end.iso,
                            'lst': lst,
                            'moon_sep': moon_sep,
                            'moon_illum': moon_illum,
                            'moon_alt': moon_alt,
                            'teff': teff,
                            'rot_start': rot_start,
                            'rot_end': rot_end,
                            'altitude': altaz.alt.deg,
                            'airmass': altaz.secz,
                            'exptime': 900, # 15 min assumption
                            'ra': r_target['target'].coord.ra.deg,
                            'dec': r_target['target'].coord.dec.deg,
                            'note': 'Manual'
                        })
                        if r_target['target'].coord not in observed_history:
                            observed_history.append(r_target['target'].coord)
                    
                    current_time = r_end # Advance current_time to the end of the manual block
                    current_pointing = r_target['target'].coord
                    # Update rotator angle (rot was calculated for the last frame mid-time, roughly correct for end state)
                    current_rotator_angle = rot_end
                
                # Case 2: Gap is too short for any observation, advance current_time past the gap
                elif (next_reservation_start - current_time) < (min_overhead + manual_readout_exptime):
                    current_time = next_reservation_start
                
                # Case 3: There is a valid gap to fill with greedy targets
                else:
                    best_candidate_for_timeslot = None
                    
                    moon_coord = get_body('moon', current_time, location=observer.location)
                    moon_altaz = observer.altaz(current_time, moon_coord)
                    sun = get_body('sun', current_time, location=observer.location)
                    moon_phase = moon_coord.separation(sun, origin_mismatch="ignore")

                    # Current telescope state for slew calc
                    if current_pointing is not None:
                        cur_altaz_coord = SkyCoord(current_pointing)
                        cur_altaz = observer.altaz(current_time, cur_altaz_coord)
                    else:
                        cur_altaz = None

                    # Iterate priorities
                    for priority in greedy_priorities:
                        if verbose:
                            print(f"[Verbose] Checking Priority {priority}")

                        # Collect targets to check
                        targets_to_check = [t for t in all_targets.values() if not t['observed'] and t['priority'] == priority]
                        
                        if not targets_to_check:
                            continue

                        # Prepare args for worker
                        worker_args = []
                        for t in targets_to_check:
                            worker_args.append((
                                t, current_time, current_pointing, cur_altaz, current_rotator_angle, 
                                next_reservation_start, observer, slew_params, min_overhead, 
                                max_airmass, rot_min, rot_max, min_teff, 
                                moon_coord, moon_altaz, moon_phase, mbm, verbose
                            ))
                        
                        # Run parallel check
                        # Using chunksize to improve performance for many small tasks
                        results = executor.map(check_visibility_worker, worker_args, chunksize=20)
                        
                        visible_candidates = [r for r in results if r is not None]
                        
                        if not visible_candidates:
                            continue

                        # Overlap Logic
                        overlapping_candidates = []
                        if observed_history:
                            for cand in visible_candidates:
                                is_overlapping = False
                                for hist_coord in observed_history:
                                    if cand['coord'].separation(hist_coord).deg < overlap_sep:
                                        is_overlapping = True
                                        break
                                if is_overlapping:
                                    overlapping_candidates.append(cand)
                        
                        if overlapping_candidates:
                            best_candidate_for_timeslot = max(overlapping_candidates, key=lambda x: x['teff'])
                            print(f"  [Overlap][P{priority}] Selected {best_candidate_for_timeslot['info']['id']} (teff: {best_candidate_for_timeslot['teff']:.2f})")
                        else:
                            best_score = -np.inf
                            for cand in visible_candidates:
                                if current_pointing is None:
                                    slew_dist = 0
                                else:
                                    slew_dist = current_pointing.separation(cand['coord']).deg
                                score = cand['alt'] - slew_penalty * slew_dist
                                
                                if score > best_score:
                                    best_score = score
                                    best_candidate_for_timeslot = cand
                            
                            if best_candidate_for_timeslot:
                                 print(f"  [Score][P{priority}] Selected {best_candidate_for_timeslot['info']['id']} (Alt: {best_candidate_for_timeslot['alt']:.1f})")

                        if best_candidate_for_timeslot:
                            break # Found best in this priority                
                    
                    if best_candidate_for_timeslot:
                        # Case 3a: An auto target was successfully scheduled
                        best_target = best_candidate_for_timeslot['info']
                        best_alt = best_candidate_for_timeslot['alt']
                        best_airmass = best_candidate_for_timeslot['airmass']
                        final_overhead = best_candidate_for_timeslot['overhead']
                        
                        exptime = best_target['exptime'] * u.s
                        
                        # Schedule starts after overhead
                        obs_start_time = current_time + final_overhead
                        obs_end_time = obs_start_time + exptime
                        
                        # Calculate Rotator Angle (Start/End)
                        pa_start = observer.parallactic_angle(obs_start_time, best_target['target']).to(u.deg).value
                        rot_start = Angle((pa_start + best_target['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value
                        
                        pa_end = observer.parallactic_angle(obs_end_time, best_target['target']).to(u.deg).value
                        rot_end = Angle((pa_end + best_target['ppc_pa']) * u.deg).wrap_at(180 * u.deg).value

                        # Calculate LST
                        lst = observer.local_sidereal_time(obs_start_time).to_string(sep=':', precision=0)
                        
                        # Moon Stats
                        moon_coord = get_body('moon', obs_start_time, location=observer.location)
                        #moon_sep = best_target['target'].coord.separation(moon_coord).deg
                        moon_sep = moon_coord.separation(best_target['target'].coord).deg
                        moon_altaz_obj = observer.altaz(obs_start_time, moon_coord)
                        moon_alt = moon_altaz_obj.alt.deg
                        moon_illum = observer.moon_illumination(obs_start_time)
                        
                        # Teff (Already calculated as best_candidate_for_timeslot['teff'])

                        schedule.append({
                            'night': night_idx + 1,
                            'target': best_target['id'],
                            'start_time': obs_start_time.iso, # Actual observation start time
                            'end_time': obs_end_time.iso,
                            'lst': lst,
                            'moon_sep': moon_sep,
                            'moon_illum': moon_illum,
                            'moon_alt': moon_alt,
                            'teff': best_candidate_for_timeslot['teff'],
                            'rot_start': rot_start,
                            'rot_end': rot_end,
                            'altitude': best_alt,
                            'airmass': best_airmass,
                            'exptime': best_target['exptime'],
                            'ra': best_target['target'].coord.ra.deg,
                            'dec': best_target['target'].coord.dec.deg,
                            'slew_time': final_overhead.to(u.s).value, # Log slew time as seconds
                            'note': 'Auto'
                        })
                        
                        # Update the master target list, as best_target is a copy from the worker
                        all_targets[best_target['id']]['observed'] = True
                        
                        best_target['observed'] = True
                        current_pointing = best_target['target'].coord
                        current_rotator_angle = best_candidate_for_timeslot['rotator_angle']
                        current_time = obs_end_time # Advance current_time to the end of observation
                        observed_history.append(current_pointing)
                    else:
                        # Case 3b: No suitable auto target found in the gap, advance current_time by a step
                        current_time += min_overhead / 2 # Advance by a smaller step, e.g., 2.5 min, to find next opportunity
    finally:
        executor.shutdown()
                
    return schedule

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plan PFS observations.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output for debugging.")
    parser.add_argument('--config', type=str, default='obs_config.yaml', help="Path to configuration file.")
    args = parser.parse_args()

    observer = setup_observer()
    
    # Load config
    config = load_config(args.config)
    
    # Read priorities
    priorities = read_priorities('targets/CO/ppcList.ecsv')
    
    # Load targets
    print("Loading targets...")
    all_targets = load_all_targets_from_ppcList(priorities)
    #xsall_targets = load_all_targets(priorities)
    print(f"Loaded {len(all_targets)} targets.")
    
    # Load manual schedule
    print("Loading manual schedule...")
    manual_schedule = load_manual_schedule('manual_allocation.csv')
    
    # Load dates
    nights = read_obsdates('obsdates_2026Jan.txt', observer)
    print(f"Loaded {len(nights)} nights.")
    
    # Run scheduler
    schedule = run_scheduler(observer, all_targets, manual_schedule, nights, config, verbose=args.verbose)
    
    # Save to CSV
    if schedule:
        keys = schedule[0].keys()
        with open('observation_schedule.csv', 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(schedule)
        print(f"\nSchedule saved to observation_schedule.csv with {len(schedule)} observations.")
        print("To generate plots from the schedule, run: python plot_schedule.py")
    else:
        print("\nNo observations scheduled.")

if __name__ == "__main__":
    main()
