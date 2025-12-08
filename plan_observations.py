import csv
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, get_body, SkyCoord, Angle
from astropy.time import Time
from astroplan import FixedTarget
import warnings
from obs_utils import setup_observer, read_obsdates, read_priorities
import datetime
import yaml

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
    adj_window = config['scheduling']['adjacency_bonus_window_sec']
    adj_bonus = config['scheduling']['adjacency_bonus_score']
    
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
        
        if is_valid:
            valid_times.append(t_start)

    if not valid_times:
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
            continue # Skip this slot as it's outside the allowed rotator angle range
            
        # Calculate altitude at mid-point
        alt = observer.altaz(t_mid, target_info['target']).alt.deg
        
        # Adjacency Bonus
        bonus = 0
        for b_start, b_end in busy_slots:
            if abs((t_start - b_end).sec) < adj_window or abs((t_end - b_start).sec) < adj_window:
                bonus = adj_bonus # Massive bonus to force adjacency
                break
        
        score = alt + bonus
        
        # Only consider valid altitudes (and prefer higher score)
        if alt > 0 and score > best_score:
            best_score = score
            best_time = t_start
            
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
            
            for req in sorted(manual_requests, key=lambda x: x['ppc_code']): # Sort for consistent results
                ppc_code = req['ppc_code']
                nframes = req['nframes']
                
                if ppc_code not in all_targets:
                    print(f"  [Error] Manual target {ppc_code} not found in database.")
                    continue
                
                target_info = all_targets[ppc_code]
                total_duration = nframes * manual_block_len
                
                # Current busy slots from reservations
                busy = [(r[0], r[1]) for r in reservations]
                
                # Align search grid to observation block length (20 min) to prevent fragmentation
                start_slot = find_optimal_slot(observer, target_info, total_duration, start_time, end_time, busy, manual_block_len, target_info['ppc_pa'], config)
                
                if start_slot:
                    end_slot = start_slot + total_duration
                    reservations.append((start_slot, end_slot, target_info, nframes))
                    mid_alt = observer.altaz(start_slot + total_duration/2, target_info['target']).alt.deg
                    print(f"    - Scheduled {ppc_code} ({nframes} frames) at {start_slot.iso} (Avg Alt: {mid_alt:.1f})")
                    target_info['observed'] = True # Mark as observed
                else:
                    print(f"    - [Warning] Could not find slot for {ppc_code}!")

        # Sort reservations by start time
        reservations.sort(key=lambda x: x[0])
        
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

                    visible_candidates = []
                    # all_targets is a dict, iterate values
                    for t in all_targets.values():
                        if t['observed'] or t['priority'] != priority:
                            continue
                        
                        # Calculate potential overhead including slew
                        # Need target AltAz and Rotator at start time to estimate slew
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
                        
                        if verbose:
                            print(f"[Verbose-Overhead] Target {t['id']}: Calculated overhead = {overhead.to(u.s).value:.2f} s")
                            
                        # Check if fits in gap
                        req_duration = t['exptime']*u.s + overhead
                        if current_time + req_duration > next_reservation_start:
                            continue
                        
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
                                if verbose:
                                    print(f"[Verbose]   - {t['id']}: Rejected (rotator angle {rotator_angle_obs_mid:.1f} out of bounds) at {current_time.iso}")
                                continue
                                
                            teff = calculate_teff(observer, t['target'].coord, alt, airmass, moon_coord, moon_altaz, moon_phase, mbm)
                            if teff > min_teff:
                                visible_candidates.append({
                                    'info': t,
                                    'alt': alt,
                                    'airmass': airmass,
                                    'coord': t['target'].coord,
                                    'teff': teff,
                                    'rotator_angle': rotator_angle_obs_mid, # Use recalculated rotator angle at mid-point
                                    'overhead': overhead # Store calculated overhead
                                })
                    
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
                    
                    best_target['observed'] = True
                    current_pointing = best_target['target'].coord
                    current_rotator_angle = best_candidate_for_timeslot['rotator_angle']
                    current_time = obs_end_time # Advance current_time to the end of observation
                    observed_history.append(current_pointing)
                else:
                    # Case 3b: No suitable auto target found in the gap, advance current_time by a step
                    current_time += min_overhead / 2 # Advance by a smaller step, e.g., 2.5 min, to find next opportunity
                
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
    all_targets = load_all_targets(priorities)
    print(f"Loaded {len(all_targets)} targets.")
    
    # Load manual schedule
    print("Loading manual schedule...")
    manual_schedule = load_manual_schedule('manual_allocation.csv')
    
    # Load dates
    nights = read_obsdates('obsdates_2025Nov.txt', observer, skip_days=8)
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
