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

## File Structure

*   `plan_observations.py`: Main scheduling logic.
*   `plot_schedule.py`: Visualization tools.
*   `obs_utils.py`: Utility functions for observer setup and file I/O.
*   `Moon.py`: Moon brightness model.
*   `slew.py`: Slew time calculations.
*   `obs_config.yaml`: Configuration file.
*   `requirements.txt`: Python dependencies.

## License

[Insert License Information Here]
