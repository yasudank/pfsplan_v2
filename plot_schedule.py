# plot_schedule.py
import csv
import numpy as np
from astropy.time import Time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import warnings

from obs_utils import setup_observer, read_targets, read_obsdates, read_priorities

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def read_schedule(filename):
    """
    Read schedule from CSV file.
    """
    schedule = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings back to numbers
                row['night'] = int(row['night'])
                row['altitude'] = float(row['altitude'])
                row['airmass'] = float(row['airmass'])
                row['exptime'] = float(row['exptime'])
                row['ra'] = float(row['ra'])
                row['dec'] = float(row['dec'])
                if 'rotator_angle' in row:
                    row['rotator_angle'] = float(row['rotator_angle'])
                # note column is string, so no conversion needed
                schedule.append(row)
    except FileNotFoundError:
        print(f"Error: Schedule file not found at {filename}")
        return None
    return schedule

def plot_altitude_time(schedule, nights):
    """
    Plot Altitude vs Time for the schedule.
    One panel per night.
    Common x-axis range (17:00 to 07:00 HST).
    Mark astronomical twilight start/end.
    Differentiate Manual vs Auto targets.
    """
    import matplotlib.pyplot as plt
    from astropy.time import Time
    import matplotlib.dates as mdates
    
    print("Generating Altitude vs Time plot...")
    
    n_nights = len(nights)
    
    fig, axes = plt.subplots(n_nights, 1, figsize=(12, 4 * n_nights), sharex=False)
    if n_nights == 1:
        axes = [axes]
    
    colors = plt.cm.jet(np.linspace(0, 1, n_nights))
    
    for i, (start_utc, end_utc) in enumerate(nights):
        night_idx = i + 1
        ax = axes[i]
        
        # Twilight times in HST
        start_hst = start_utc.datetime - datetime.timedelta(hours=10)
        end_hst = end_utc.datetime - datetime.timedelta(hours=10)
        
        # Plot twilight lines
        ax.axvline(start_hst, color='red', linestyle='--', alpha=0.7, label='Twilight')
        ax.axvline(end_hst, color='red', linestyle='--', alpha=0.7)
        
        # Filter observations for this night
        night_obs = [s for s in schedule if s['night'] == night_idx]
        
        if night_obs:
            # Separate Manual and Auto
            manual_obs = [s for s in night_obs if s.get('note') == 'Manual']
            auto_obs = [s for s in night_obs if s.get('note') != 'Manual']

            # Plot Auto
            if auto_obs:
                times_auto = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in auto_obs]
                alts_auto = [s['altitude'] for s in auto_obs]
                ax.scatter(times_auto, alts_auto, label=f"Auto", color=colors[i], s=20, marker='o')

            # Plot Manual
            if manual_obs:
                times_man = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in manual_obs]
                alts_man = [s['altitude'] for s in manual_obs]
                ax.scatter(times_man, alts_man, label=f"Manual", color='red', s=60, marker='*', edgecolors='black')
        
        ax.set_ylabel("Altitude (deg)")
        ax.set_title(f"Night {night_idx} (HST)")
        ax.grid(True, alpha=0.3)
        
        # Set common x-axis limits: 17:00 previous day to 07:00 current day
        # Anchor to the twilight start date
        # If start_hst is early morning (e.g. < 15:00), it belongs to the previous date's night
        anchor_date = start_hst.date()
        if start_hst.hour < 15:
            anchor_date -= datetime.timedelta(days=1)
            
        xlim_start = datetime.datetime.combine(anchor_date, datetime.time(17, 0))
        xlim_end = xlim_start + datetime.timedelta(hours=14) # 07:00 next day
        
        ax.set_xlim(xlim_start, xlim_end)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Add legend
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel("Time (HST)")
    plt.tight_layout()
    
    plt.savefig("altitude_vs_time.png")
    plt.close(fig)
    print("Saved altitude_vs_time.png")

def plot_rotator_angle_time(schedule, nights):
    """
    Plot Rotator Angle vs Time for the schedule.
    """
    import matplotlib.pyplot as plt
    from astropy.time import Time
    import matplotlib.dates as mdates
    
    print("Generating Rotator Angle vs Time plot...")
    
    n_nights = len(nights)
    
    fig, axes = plt.subplots(n_nights, 1, figsize=(12, 4 * n_nights), sharex=False)
    if n_nights == 1:
        axes = [axes]
    
    colors = plt.cm.jet(np.linspace(0, 1, n_nights))
    
    for i, (start_utc, end_utc) in enumerate(nights):
        night_idx = i + 1
        ax = axes[i]
        
        # Twilight times in HST
        start_hst = start_utc.datetime - datetime.timedelta(hours=10)
        end_hst = end_utc.datetime - datetime.timedelta(hours=10)
        
        # Plot twilight lines
        ax.axvline(start_hst, color='red', linestyle='--', alpha=0.7, label='Twilight')
        ax.axvline(end_hst, color='red', linestyle='--', alpha=0.7)
        
        # Filter observations for this night
        night_obs = [s for s in schedule if s['night'] == night_idx]
        
        if night_obs:
            # Separate Manual and Auto
            manual_obs = [s for s in night_obs if s.get('note') == 'Manual']
            auto_obs = [s for s in night_obs if s.get('note') != 'Manual']

            # Plot Auto
            if auto_obs:
                times_auto = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in auto_obs]
                rots_auto = [s.get('rotator_angle', 0) for s in auto_obs]
                ax.scatter(times_auto, rots_auto, label=f"Auto", color=colors[i], s=20, marker='o')

            # Plot Manual
            if manual_obs:
                times_man = [Time(s['start_time']).datetime - datetime.timedelta(hours=10) for s in manual_obs]
                rots_man = [s.get('rotator_angle', 0) for s in manual_obs]
                ax.scatter(times_man, rots_man, label=f"Manual", color='red', s=60, marker='*', edgecolors='black')
        
        ax.set_ylabel("Rotator Angle (deg)")
        ax.set_title(f"Night {night_idx} (HST)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-180, 180)
        
        # Set common x-axis limits
        anchor_date = start_hst.date()
        if start_hst.hour < 15:
            anchor_date -= datetime.timedelta(days=1)
            
        xlim_start = datetime.datetime.combine(anchor_date, datetime.time(17, 0))
        xlim_end = xlim_start + datetime.timedelta(hours=14) 
        
        ax.set_xlim(xlim_start, xlim_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel("Time (HST)")
    plt.tight_layout()
    
    plt.savefig("rotator_angle_vs_time.png")
    plt.close(fig)
    print("Saved rotator_angle_vs_time.png")

def plot_sky_coverage(schedule, all_targets):
    """
    Plot Sky Coverage.
    Two panels:
    1. RA -30 to 40 (Autumn targets)
    2. RA 125 to 155 (Spring targets)
    
    Plots:
    - Scheduled targets: Filled circles with specific color
    - Unobserved targets: Unfilled circles with specific color
    
    Colors:
    - SSP_CO: blueviolet
    - SSP_GA: yellowgreen
    - SSP_GE: orangered
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    
    print("Generating Sky Coverage plot...")
    
    try:
        scheduled_ids = set(s['target'] for s in schedule)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Helper for color
        def get_color(tid):
            if tid.startswith('SSP_CO'):
                return 'blueviolet'
            elif tid.startswith('SSP_GA'):
                return 'yellowgreen'
            elif tid.startswith('SSP_GE'):
                return 'orangered'
            return 'grey'

        for t in all_targets:
            tid = t['id']
            ra = t['target'].coord.ra.deg
            dec = t['target'].coord.dec.deg
            
            is_scheduled = tid in scheduled_ids
            color = get_color(tid)
            
            if is_scheduled:
                facecolor = color
                edgecolor = color
                alpha = 0.6
                fill = True
                zorder = 10
            else:
                facecolor = 'none'
                edgecolor = color
                alpha = 0.4
                fill = False
                zorder = 1
            
            # Panel 1: RA -30 to 40
            # Shift RA: if RA > 180, RA -= 360
            ra_shifted = ra - 360 if ra > 180 else ra
            
            if -35 < ra_shifted < 45: # Loose bounds for inclusion
                circle = Circle((ra_shifted, dec), 0.7, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, fill=fill, zorder=zorder)
                ax1.add_patch(circle)
            
            # Panel 2: RA 125 to 155
            if 120 < ra < 160: # Loose bounds
                circle = Circle((ra, dec), 0.7, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, fill=fill, zorder=zorder)
                ax2.add_patch(circle)
            
        ax1.set_xlim(-30, 40)
        ax1.set_ylim(-6.5, 5) 
        ax1.set_xlabel("RA (deg)")
        ax1.set_ylabel("Dec (deg)")
        ax1.set_title("Sky Coverage (RA -30 to 40)")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.invert_xaxis()
        
        ax2.set_xlim(125, 155)
        ax2.set_ylim(-6.5, 5)
        ax2.set_xlabel("RA (deg)")
        ax2.set_ylabel("Dec (deg)")
        ax2.set_title("Sky Coverage (RA 125 to 155)")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.invert_xaxis()

        # Custom Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blueviolet', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='yellowgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='orangered', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig("sky_coverage.png")
        plt.close(fig)
        print("Saved sky_coverage.png")
        
    except Exception as e:
        print(f"Error generating sky coverage plot: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

def plot_sky_coverage_mollweide(schedule, all_targets):
    """
    Plot Sky Coverage for the entire sky using Mollweide projection.
    Colors:
    - SSP_CO: blueviolet
    - SSP_GA: yellowgreen
    - SSP_GE: orangered
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    
    print("Generating Sky Coverage (Mollweide) plot...")
    
    try:
        if schedule:
            scheduled_ids = set(s['target'] for s in schedule)
        else:
            scheduled_ids = set()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='mollweide')
        
        # Helper for color
        def get_category(tid):
            if tid.startswith('SSP_CO'):
                return 'SSP_CO', 'blueviolet'
            elif tid.startswith('SSP_GA'):
                return 'SSP_GA', 'yellowgreen'
            elif tid.startswith('SSP_GE'):
                return 'SSP_GE', 'orangered'
            return 'Other', 'grey'

        # Data structure to hold coordinates for each category and status
        # keys: (category, is_scheduled)
        # values: {'ra': [], 'dec': [], 'color': str}
        data = {}

        for t in all_targets:
            tid = t['id']
            ra_deg = t['target'].coord.ra.deg
            dec_deg = t['target'].coord.dec.deg
            
            # Coordinate transformation
            ra_rad = np.radians(ra_deg)
            if ra_rad > np.pi:
                ra_rad -= 2 * np.pi
            ra_plot = -ra_rad
            dec_plot = np.radians(dec_deg)
            
            is_scheduled = tid in scheduled_ids
            cat, color = get_category(tid)
            
            key = (cat, is_scheduled)
            if key not in data:
                data[key] = {'ra': [], 'dec': [], 'color': color}
            
            data[key]['ra'].append(ra_plot)
            data[key]['dec'].append(dec_plot)
            
        # Plotting
        for (cat, is_scheduled), val in data.items():
            if not val['ra']:
                continue
                
            color = val['color']
            if is_scheduled:
                ax.scatter(val['ra'], val['dec'], c=color, edgecolors=color, alpha=0.7, s=20, label=f"{cat} (Sched)")
            else:
                # Unobserved: open circles
                ax.scatter(val['ra'], val['dec'], facecolors='none', edgecolors=color, alpha=0.4, s=15, label=f"{cat} (Unobs)")

        ax.grid(True)
        ax.set_title("Sky Coverage (Mollweide, RA 0 at Center)")

        # Customizing x-axis labels
        tick_locations = np.radians(np.arange(-150, 180, 30))
        ax.set_xticks(tick_locations)
        
        tick_labels = []
        for x in tick_locations:
            ra_val = np.degrees(-x)
            if ra_val < 0:
                ra_val += 360
            tick_labels.append(f"{int(round(ra_val))}°")
            
        ax.set_xticklabels(tick_labels)
        
        # Custom Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blueviolet', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='yellowgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='orangered', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig("sky_coverage_mollweide.png")
        plt.close(fig)
        print("Saved sky_coverage_mollweide.png")
        
    except Exception as e:
        print(f"Error generating sky coverage mollweide plot: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)
        
    except Exception as e:
        print(f"Error generating sky coverage mollweide plot: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

def animate_sky_coverage_progress(schedule, all_targets):
    """
    Animate Sky Coverage progress (cumulative) over the nights.
    Saves as 'sky_coverage_progress.gif'.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from matplotlib.animation import FuncAnimation
    
    print("Generating Sky Coverage Progress animation...")
    
    if not schedule:
        print("No schedule to animate.")
        return

    nights = sorted(list(set(s['night'] for s in schedule)))
    if not nights:
        return
    max_night = max(nights)
    
    # Pre-calculate earliest night for each target
    target_first_night = {}
    for s in schedule:
        tid = s['target']
        n = s['night']
        if tid not in target_first_night or n < target_first_night[tid]:
            target_first_night[tid] = n
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Helper for color
    def get_color(tid):
        if tid.startswith('SSP_CO'):
            return 'blueviolet'
        elif tid.startswith('SSP_GA'):
            return 'yellowgreen'
        elif tid.startswith('SSP_GE'):
            return 'orangered'
        return 'grey'
        
    patches_map = {}
    
    # Initialize plot with ALL targets as Unobserved
    for t in all_targets:
        tid = t['id']
        ra = t['target'].coord.ra.deg
        dec = t['target'].coord.dec.deg
        color = get_color(tid)
        
        patches_map[tid] = []
        
        # Panel 1: RA -30 to 40
        ra_shifted = ra - 360 if ra > 180 else ra
        if -35 < ra_shifted < 45:
            c = Circle((ra_shifted, dec), 0.7, facecolor='none', edgecolor=color, alpha=0.4, fill=False, zorder=1)
            ax1.add_patch(c)
            patches_map[tid].append(c)
            
        # Panel 2: RA 125 to 155
        if 120 < ra < 160:
            c = Circle((ra, dec), 0.7, facecolor='none', edgecolor=color, alpha=0.4, fill=False, zorder=1)
            ax2.add_patch(c)
            patches_map[tid].append(c)
            
    # Axes setup
    ax1.set_xlim(-30, 40)
    ax1.set_ylim(-6.5, 5) 
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.invert_xaxis()
    
    ax2.set_xlim(125, 155)
    ax2.set_ylim(-6.5, 5)
    ax2.set_xlabel("RA (deg)")
    ax2.set_ylabel("Dec (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.invert_xaxis()
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='SSP_CO', markerfacecolor='blueviolet', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SSP_GA', markerfacecolor='yellowgreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SSP_GE', markerfacecolor='orangered', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Scheduled (Filled)', markerfacecolor='grey', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unobserved (Open)', markerfacecolor='none', markeredgecolor='grey', markersize=10),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    def update(frame):
        # Update titles
        title_suffix = "Start" if frame == 0 else f"Night {frame}"
        ax1.set_title(f"Sky Coverage (RA -30 to 40) - {title_suffix}")
        ax2.set_title(f"Sky Coverage (RA 125 to 155) - {title_suffix}")
        
        for tid, patches in patches_map.items():
            color = get_color(tid)
            # Check if observed on or before this frame (night)
            # frame 0 means no observations shown as filled yet
            is_observed = (tid in target_first_night) and (target_first_night[tid] <= frame) and (frame > 0)
            
            for p in patches:
                if is_observed:
                    p.set_facecolor(color)
                    p.set_alpha(0.6)
                    p.set_fill(True)
                    p.set_zorder(10)
                else:
                    p.set_facecolor('none')
                    p.set_alpha(0.4)
                    p.set_fill(False)
                    p.set_zorder(1)
        return []

    # Create animation from frame 0 (empty) to max_night
    frames = range(0, max_night + 1)
    anim = FuncAnimation(fig, update, frames=frames, interval=800)
    
    try:
        anim.save('sky_coverage_progress.gif', writer='pillow', fps=2)
        print("Saved sky_coverage_progress.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")
        import traceback
        traceback.print_exc()

    plt.close(fig)

def main():
    observer = setup_observer()
    
    # Read priorities first
    priorities = read_priorities('targets/CO/ppcList.ecsv')
    if not priorities:
        print("Warning: Could not read priorities. Proceeding with default priority for all targets.")
    
    all_targets = []
    target_files = ['CO_summary_reconfigure.csv', 'GA_summary_reconfigure.csv', 'GE_summary_reconfigure.csv']
    
    for fname in target_files:
        try:
            targets = read_targets(fname, priorities)
            all_targets.extend(targets)
            print(f"Loaded {len(targets)} targets from {fname}.")
        except FileNotFoundError:
            print(f"Warning: Target file '{fname}' not found.")
    
    print(f"Total targets loaded: {len(all_targets)}")

    schedule = read_schedule('observation_schedule.csv')
    if schedule is None:
        return
    
    if not schedule:
        print("Schedule is empty. Generating empty sky coverage plot.")
        plot_sky_coverage(schedule, all_targets)
        plot_sky_coverage_mollweide(schedule, all_targets)
        print("Altitude vs time plot not generated as there are no observations.")
        return

    print(f"Loaded {len(schedule)} observations from schedule.")

    num_nights = max(s['night'] for s in schedule)
        
    try:
        nights = read_obsdates('obsdates_2025Nov.txt', observer, skip_days=8)
        print(f"Loaded {len(nights)} observation windows.")
    except FileNotFoundError:
        print("Error: Observation dates file 'obsdates_2025Nov.txt' not found.")
        print("Cannot generate altitude vs time plot.")
        nights = []

    if nights:
        plot_altitude_time(schedule, nights)
        plot_rotator_angle_time(schedule, nights)
    plot_sky_coverage(schedule, all_targets)
    plot_sky_coverage_mollweide(schedule, all_targets)
    animate_sky_coverage_progress(schedule, all_targets)

if __name__ == '__main__':
    main()
