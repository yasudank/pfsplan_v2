# PFS Observation Planner (v2)

This project provides a suite of Python tools for planning, scheduling, and visualizing observations for the Prime Focus Spectrograph (PFS) at the Subaru Telescope. It automates the scheduling process based on target visibility, moon brightness, and telescope constraints, and generates detailed visual reports of the observation plan.

## Features

- **Automated Scheduling**: Optimizes observation times based on target priority, visibility, and constraints (airmass, altitude, moon distance, etc.).
- **Moon Brightness Modeling**: Includes a detailed model for calculating sky brightness contributions from the moon in various bands (`g`, `r`, `i`, `z`, `y`).
- **Slew Time Calculation**: Estimates slew times for azimuth, elevation, and rotator movements to ensure realistic scheduling.
- **Visualization**: Generates comprehensive plots, including:
  - Altitude vs. Time
  - Rotator Angle vs. Time
  - Sky Coverage (Single panel with dynamic RA range)
  - Animated Sky Coverage Progress
  - Observation Counts per Target (Stacked bars for GA/GE groups)
- **PDF Reporting**: Automatically generates a structured PDF report with all plots and tabular schedules.
- **Configurable**: Highly customizable parameters via `obs_config.yaml`.

## Prerequisites

- Python 3.10 or higher
- Standard scientific Python stack (`numpy`, `pandas`, `matplotlib`)
- Astronomy packages (`astropy`, `astroplan`, `pyerfa`)
- PDF Generation: `reportlab`

## Installation

1.  Clone the repository or extract the source code.
2.  Install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

### `obs_config.yaml`

The main configuration file `obs_config.yaml` allows you to tweak various system parameters:

*   **`slew`**: Slew speeds for azimuth, elevation, and rotator.
*   **`constraints`**: 
    *   `max_airmass`: Standard limit (e.g., 1.6).
    *   `max_airmass_relaxed`: Limit for a restricted number of frames (e.g., 2.2).
    *   `max_relaxed_count`: Max frames allowed to use the relaxed airmass per night.
    *   `max_altitude`: Upper altitude limit (e.g., 75.0°).
    *   `rotator_min/max`: Instrument rotator angle limits.
*   **`scheduling`**: 
    *   `group_all_manual_daily`: If true, treats all manual targets in a day as one block.
    *   `group_all_exclude_dates`: List of dates to skip the global daily grouping.

## Usage

### 1. Generate Schedule

Run the main planning script to generate the observation schedule:

```bash
python plan_observations.py [options]
```

**Options:**
*   `--config <file>`: Path to configuration file (default: `obs_config.yaml`).
*   `--manual <file>`: Path to manual allocation CSV (default: `manual_allocation_2026Mar.csv`).
*   `--obsdates <file>`: Path to observation dates text file (default: `obsdates_2026Mar.txt`).
*   `--group-all`: Treat all manual targets for a day as one block, regardless of prefix.
*   `--exclude-dates <dates>`: Comma-separated list of dates (YYYY-MM-DD) to exclude from `--group-all`.
*   `--verbose-slew`: Enable verbose output for slew time calculation details.

### 2. Visualize Schedule

After generating the schedule, run the plotting script to create visual reports:

```bash
python plot_schedule.py
```

### 3. Generate PDF Report

Create a comprehensive PDF report combining all plots and the schedule:

```bash
python create_pdf_report.py
```

## Scheduling Algorithm

The `plan_observations.py` script employs a hybrid scheduling approach:

### 1. Manual Allocation Phase
Processes targets defined in the manual allocation CSV:
*   **Grouping**: Consecutive targets are grouped by program prefix (e.g., `SSP_GA`) or daily (if `--group-all` is used).
*   **Block Optimization**: `find_optimal_slot` calculates the best time for the *entire block* by maximizing the average altitude while ensuring every target meets its individual constraints (including the `max_altitude` and `relaxed airmass` budget).
*   **Gap Compaction**: Automatically closes small gaps (< 80 mins) between blocks to maximize night efficiency, respecting the `max_relaxed_count` quota.

### 2. Auto-Scheduling Phase (Greedy Gap Filling)
Fills remaining gaps with unscheduled targets:
*   **Parallel Processing**: Efficiently checks visibility for hundreds of candidates simultaneously.
*   **Multi-Priority Greedy**: Iterates through priorities and selects the best candidate based on altitude and slew distance.
*   **Overlap Logic**: Prioritizes fields spatially close to recently observed targets to maintain survey continuity.

## File Structure

*   `plan_observations.py`: Main scheduling logic.
*   `plot_schedule.py`: Visualization and plotting tools.
*   `create_pdf_report.py`: PDF report generation.
*   `obs_utils.py`: Shared utility functions (I/O, observer setup).
*   `obs_config.yaml`: Central configuration.
*   `Moon.py` (internal): Moon brightness modeling logic.

## License

[Insert License Information Here]
