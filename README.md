# PFS Observation Planner (v2)

This project provides a suite of Python tools for planning, scheduling, and visualizing observations for the Prime Focus Spectrograph (PFS) at the Subaru Telescope. It automates the scheduling process based on target visibility, moon brightness, and telescope constraints, and generates detailed visual reports of the observation plan.

## Features

- **Automated Scheduling**: Optimizes observation times based on target priority, visibility, and constraints (airmass, moon distance, etc.).
- **Moon Brightness Modeling**: Includes a detailed model for calculating sky brightness contributions from the moon in various bands (`g`, `r`, `i`, `z`, `y`).
- **Slew Time Calculation**: Estimates slew times for azimuth, elevation, and rotator movements to ensure realistic scheduling.
- **Visualization**: Generates comprehensive plots, including:
  - Altitude vs. Time
  - Rotator Angle vs. Time
  - Sky Coverage (Static and Animated)
- **Configurable**: Highly customizable parameters via `obs_config.yaml`.

## Prerequisites

- Python 3.10 or higher
- Standard scientific Python stack (`numpy`, `pandas`, `matplotlib`)
- Astronomy packages (`astropy`, `astroplan`, `pyerfa`)

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
*   **`constraints`**: Observational limits such as maximum airmass, minimum altitude, and rotator limits.
*   **`scheduling`**: Parameters for the scheduler, including overhead times and bonuses for overlapping fields.

### Input Files

*   **Observation Dates**: Defined in text files like `obsdates_2025Nov.txt` (format: list of dates).
*   **Target Lists**: CSV files containing target information (e.g., `CO_summary_reconfigure.csv`, `GA_summary_reconfigure.csv`).

## Usage

### 1. Generate Schedule

Run the main planning script to generate the observation schedule:

```bash
python plan_observations.py
```

This script reads the configuration and target lists, processes the logic, and outputs the schedule to `observation_schedule.csv`.

### 2. Visualize Schedule

After generating the schedule, run the plotting script to create visual reports:

```bash
python plot_schedule.py
```

This will generate several image files in the current directory, including:
*   `altitude_vs_time.png`
*   `rotator_angle_vs_time.png`
*   `sky_coverage.png`
*   `sky_coverage_progress.gif`

### 3. Generate PDF Report

After generating the schedule and plots, you can create a comprehensive PDF report:

```bash
python create_pdf_report.py
```

This will generate `schedule_report.pdf`, which includes all generated plots and a detailed tabular schedule for each night, highlighting any observations that violate configured constraints.

## Scheduling Algorithm

The `plan_observations.py` script employs a hybrid scheduling approach, ensuring high-priority manual targets are fixed while efficiently filling gaps with a greedy optimization strategy.

### 1. Manual Allocation Phase
First, the scheduler processes targets defined in `manual_allocation.csv`:
*   **Prioritization**: Enforces project-specific ordering (e.g., GE targets before GA).
*   **Smart Grouping**: Spatially adjacent targets (separation < 5°) are grouped into contiguous blocks.
*   **Slot Optimization**: Finds optimal time slots based on altitude and rotator constraints. It includes logic to "compact" the schedule, shifting blocks to close unusable gaps (< 80 mins) or align with the night's start/end.

### 2. Auto-Scheduling Phase (Greedy Gap Filling)
Remaining time slots are filled dynamically:
*   **Constraint Checking**: Calculates visibility for all candidates using parallel processing, checking airmass, rotator limits, and moon constraints (brightness/distance).
*   **Scoring & Selection**:
    1.  **Overlap**: Prioritizes targets overlapping with recent observations to maximize survey continuity.
    2.  **Efficiency Score**: For non-overlapping targets, selects based on `Altitude - (Slew Penalty * Distance)`, balancing airmass against slew time.

## File Structure

*   `plan_observations.py`: Main scheduling logic.
*   `plot_schedule.py`: Visualization tools.
*   `create_pdf_report.py`: Generates a comprehensive PDF report of the schedule and plots.
*   `schedule_report.pdf`: The output PDF report.
*   `obs_utils.py`: Utility functions for observer setup and file I/O.
*   `Moon.py`: Moon brightness model.
*   `slew.py`: Slew time calculations.
*   `obs_config.yaml`: Configuration file.
*   `requirements.txt`: Python dependencies.

## License

[Insert License Information Here]
