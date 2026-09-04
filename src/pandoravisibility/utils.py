"""
Extended utilities for pandoravisibility analysis.

This module provides higher-level analysis functions that build on the core
Visibility class for mission planning and target analysis.
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .visibility import Visibility

__all__ = [
    "analyze_yearly_visibility",
    "find_continuous_periods",
    "calculate_visibility_statistics",
    "find_optimal_observation_windows",
    "export_visibility_periods",
    "analyze_target_yearly_visibility",
    "plot_yearly_visibility",
    "plot_visibility_summary",
    "plot_observation_windows",
]


def analyze_yearly_visibility(
    tle_line1,
    tle_line2,
    target_coord,
    start_time=None,
    duration_days=365,
    time_resolution_hours=1,
    visibility_threshold_hours=24,
    verbose=False,
):
    """
    Analyze when a target will be continuously visible over a year.

    Parameters:
    -----------
    tle_line1, tle_line2 : str
        TLE lines for satellite
    target_coord : SkyCoord
        Target coordinates
    start_time : Time, optional
        Start time for analysis (default: Time.now())
    duration_days : int
        Duration to analyze in days (default: 365)
    time_resolution_hours : float
        Time resolution for visibility checks in hours (default: 1)
    visibility_threshold_hours : float
        Minimum continuous visibility duration in hours (default: 24)
    verbose : bool
        Print progress information

    Returns:
    --------
    dict
        Dictionary with visibility analysis results
    """

    if start_time is None:
        start_time = Time.now()
    elif not isinstance(start_time, Time):
        # Allow string input and convert to Time
        start_time = Time(start_time)

    if verbose:
        print("Analyzing yearly visibility for target:")
        print(f"  RA: {target_coord.ra:.4f}")
        print(f"  DEC: {target_coord.dec:.4f}")
        print(f"  Start time: {start_time.iso}")
        print(f"  Duration: {duration_days} days")
        print(f"  Time resolution: {time_resolution_hours} hours")

    # Create time array using astropy
    total_hours = duration_days * 24
    n_points = int(total_hours / time_resolution_hours)

    time_deltas = np.arange(0, total_hours, time_resolution_hours) * u.hour
    times = start_time + time_deltas

    if verbose:
        print(f"  Checking {n_points} time points over {duration_days} days")
        print(f"  End time: {times[-1].iso}")

    # Initialize visibility calculator
    vis = Visibility(tle_line1, tle_line2)

    # Calculate visibility for all times
    if verbose:
        print("  Computing visibility...")

    visibility_results = np.asarray(
        vis.get_visibility(target_coord, times)["visible"]
    )

    # Find continuous visibility periods
    continuous_periods = find_continuous_periods(
        times, visibility_results, visibility_threshold_hours
    )

    # Calculate statistics
    stats = calculate_visibility_statistics(
        times, visibility_results, continuous_periods, visibility_threshold_hours
    )

    if verbose:
        print(f"  Found {len(continuous_periods)} continuous visibility periods")
        print(
            f"  Total visible time: {stats['total_visible_hours']:.1f} hours ({stats['visibility_percentage']:.1f}%)"
        )

    results = {
        "times": times,
        "visibility": visibility_results,
        "continuous_periods": continuous_periods,
        "statistics": stats,
        "target_coord": target_coord,
        "analysis_params": {
            "start_time": start_time,
            "duration_days": duration_days,
            "time_resolution_hours": time_resolution_hours,
            "visibility_threshold_hours": visibility_threshold_hours,
        },
    }

    return results


def find_continuous_periods(times, visibility, min_duration_hours):
    """
    Find periods of continuous visibility.

    Parameters:
    -----------
    times : Time array
        Time points
    visibility : bool array
        Visibility results
    min_duration_hours : float
        Minimum duration for a period to be considered

    Returns:
    --------
    list
        List of continuous visibility periods
    """
    periods = []

    # Find transitions
    visible_diff = np.diff(np.concatenate(([False], visibility, [False])).astype(int))

    # Find start and end indices of visible periods
    start_indices = np.where(visible_diff == 1)[0]
    end_indices = np.where(visible_diff == -1)[0]

    for start_idx, end_idx in zip(start_indices, end_indices):
        if end_idx > start_idx:  # Valid period
            start_time = times[start_idx]
            end_time = times[end_idx - 1]  # Last visible time
            duration = (end_time - start_time).to(u.hour)

            if duration.value >= min_duration_hours:
                periods.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": duration.value,
                        "duration_days": duration.to(u.day).value,
                        "start_index": start_idx,
                        "end_index": end_idx - 1,
                    }
                )

    return periods


def calculate_visibility_statistics(
    times, visibility, continuous_periods, min_duration_hours
):
    """Calculate visibility statistics."""

    total_time = times[-1] - times[0]
    total_time_hours = total_time.to(u.hour).value
    total_time_days = total_time.to(u.day).value

    visible_points = np.sum(visibility)
    total_points = len(visibility)

    time_resolution_hours = total_time_hours / total_points
    total_visible_hours = visible_points * time_resolution_hours
    visibility_percentage = (visible_points / total_points) * 100

    # Statistics for continuous periods
    if continuous_periods:
        period_durations = [period["duration_hours"] for period in continuous_periods]
        longest_period_hours = max(period_durations)
        average_period_hours = np.mean(period_durations)
        total_continuous_hours = sum(period_durations)
    else:
        longest_period_hours = 0
        average_period_hours = 0
        total_continuous_hours = 0

    return {
        "total_time_hours": total_time_hours,
        "total_time_days": total_time_days,
        "total_visible_hours": total_visible_hours,
        "visibility_percentage": visibility_percentage,
        "continuous_periods_count": len(continuous_periods),
        "longest_continuous_period_hours": longest_period_hours,
        "longest_continuous_period_days": longest_period_hours / 24,
        "average_continuous_period_hours": average_period_hours,
        "total_continuous_hours": total_continuous_hours,
        "continuous_visibility_percentage": (total_continuous_hours / total_time_hours)
        * 100,
        "time_resolution_hours": time_resolution_hours,
    }


def find_optimal_observation_windows(
    results, observation_duration_hours, min_gap_hours=2
):
    """
    Find optimal observation windows within continuous visibility periods.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    observation_duration_hours : float
        Required observation duration
    min_gap_hours : float
        Minimum gap between observations

    Returns:
    --------
    list
        List of optimal observation windows
    """

    observation_windows = []

    for period in results["continuous_periods"]:
        if period["duration_hours"] >= observation_duration_hours:
            # Calculate how many observations can fit
            available_time = period["duration_hours"]
            observation_plus_gap = observation_duration_hours + min_gap_hours
            n_observations = int(available_time / observation_plus_gap)

            if n_observations > 0:
                # Distribute observations evenly within the period
                period_duration = period["end_time"] - period["start_time"]
                observation_duration_astropy = observation_duration_hours * u.hour

                for i in range(n_observations):
                    # Calculate start time for this observation
                    time_offset_ratio = (i + 0.5) / n_observations
                    obs_start = (
                        period["start_time"] + time_offset_ratio * period_duration
                    )
                    obs_end = obs_start + observation_duration_astropy

                    # Make sure we don't exceed the period
                    if obs_end <= period["end_time"]:
                        observation_windows.append(
                            {
                                "start_time": obs_start,
                                "end_time": obs_end,
                                "duration_hours": observation_duration_hours,
                                "duration_days": observation_duration_hours / 24,
                                "parent_period_start": period["start_time"],
                                "parent_period_end": period["end_time"],
                                "observation_number": i + 1,
                                "total_observations_in_period": n_observations,
                            }
                        )

    return observation_windows


def export_visibility_periods(results, filename):
    """
    Export continuous visibility periods to CSV.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    filename : str
        Output CSV filename

    Returns:
    --------
    DataFrame
        Pandas DataFrame with the exported data
    """

    periods_data = []
    for i, period in enumerate(results["continuous_periods"]):
        periods_data.append(
            {
                "period_number": i + 1,
                "start_time_iso": period["start_time"].iso,
                "end_time_iso": period["end_time"].iso,
                "start_time_mjd": period["start_time"].mjd,
                "end_time_mjd": period["end_time"].mjd,
                "duration_hours": period["duration_hours"],
                "duration_days": period["duration_days"],
            }
        )

    df = pd.DataFrame(periods_data)
    df.to_csv(filename, index=False)

    print(f"Exported {len(periods_data)} continuous visibility periods to {filename}")

    return df


def analyze_target_yearly_visibility(
    target_ra, target_dec, tle1, tle2, start_time=None, **kwargs
):
    """
    Complete analysis workflow for a target.

    Parameters:
    -----------
    target_ra, target_dec : float
        Target coordinates in degrees
    tle1, tle2 : str
        TLE lines
    start_time : Time, str, or None
        Start time for analysis
    **kwargs
        Additional arguments passed to analyze_yearly_visibility

    Returns:
    --------
    dict
        Complete analysis results
    """

    print("=" * 70)
    print("YEARLY TARGET VISIBILITY ANALYSIS")
    print("=" * 70)

    # Create target coordinate
    target_coord = SkyCoord(target_ra, target_dec, frame="icrs", unit="deg")

    # Handle start_time
    if start_time is not None and not isinstance(start_time, Time):
        start_time = Time(start_time)

    # Set default parameters
    default_params = {
        "time_resolution_hours": 1,
        "visibility_threshold_hours": 24,
        "verbose": True,
    }
    default_params.update(kwargs)

    # Run yearly analysis
    results = analyze_yearly_visibility(
        tle1, tle2, target_coord, start_time=start_time, **default_params
    )

    # Print summary
    stats = results["statistics"]
    print("\nVISIBILITY SUMMARY:")
    print(
        f"• Analysis period: {results['analysis_params']['start_time'].iso} to {results['times'][-1].iso}"
    )
    print(
        f"• Total visibility: {stats['visibility_percentage']:.1f}% of analysis period"
    )
    print(
        f"• Continuous periods (≥{default_params['visibility_threshold_hours']}h): {stats['continuous_periods_count']}"
    )

    if stats["continuous_periods_count"] > 0:
        print(
            f"• Longest continuous period: {stats['longest_continuous_period_hours']:.1f} hours ({stats['longest_continuous_period_days']:.1f} days)"
        )
        print(
            f"• Average period duration: {stats['average_continuous_period_hours']:.1f} hours"
        )

    # Find optimal observation windows
    obs_windows = find_optimal_observation_windows(
        results, observation_duration_hours=6
    )
    print(f"• Optimal 6-hour observation windows: {len(obs_windows)}")

    # Export results
    filename = f"visibility_periods_ra{target_ra:.1f}_dec{target_dec:.1f}.csv"
    periods_df = export_visibility_periods(results, filename)

    return {
        "analysis": results,
        "observation_windows": obs_windows,
        "periods_dataframe": periods_df,
    }


def get_visibility_summary(results):
    """
    Get a human-readable summary of visibility analysis results.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility

    Returns:
    --------
    str
        Formatted summary string
    """
    stats = results["statistics"]
    params = results["analysis_params"]

    summary_lines = [
        "VISIBILITY ANALYSIS SUMMARY",
        "=" * 50,
        f"Target: RA {results['target_coord'].ra:.4f}, DEC {results['target_coord'].dec:.4f}",
        f"Analysis period: {params['start_time'].iso} ({params['duration_days']} days)",
        f"Time resolution: {params['time_resolution_hours']} hours",
        "",
        f"Overall Visibility: {stats['visibility_percentage']:.1f}%",
        f"Total visible time: {stats['total_visible_hours']:.1f} hours ({stats['total_visible_hours']/24:.1f} days)",
        "",
        f"Continuous Periods (≥{params['visibility_threshold_hours']}h):",
        f"  Count: {stats['continuous_periods_count']}",
        f"  Longest: {stats['longest_continuous_period_hours']:.1f} hours ({stats['longest_continuous_period_days']:.1f} days)",
        f"  Average: {stats['average_continuous_period_hours']:.1f} hours",
        f"  Total continuous time: {stats['total_continuous_hours']:.1f} hours ({stats['continuous_visibility_percentage']:.1f}%)",
    ]

    return "\n".join(summary_lines)


def plot_yearly_visibility(results, figsize=(15, 10)):
    """
    Plot yearly visibility analysis results.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    figsize : tuple
        Figure size (default: (15, 10))

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    times = results["times"]
    visibility = results["visibility"]
    continuous_periods = results["continuous_periods"]
    stats = results["statistics"]
    params = results["analysis_params"]

    # Convert astropy Time to matplotlib-compatible datetime
    times_datetime = [t.datetime for t in times]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot 1: Raw visibility
    axes[0].plot(times_datetime, visibility.astype(int), "b-", linewidth=0.5, alpha=0.7)
    axes[0].fill_between(
        times_datetime, 0, visibility.astype(int), alpha=0.3, color="blue"
    )
    axes[0].set_ylabel("Visible\n(1=Yes, 0=No)")
    axes[0].set_title(
        f"Target Visibility Over Time\n"
        f'RA: {results["target_coord"].ra:.3f}, '
        f'DEC: {results["target_coord"].dec:.3f}'
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # Plot 2: Continuous periods
    y_continuous = np.zeros_like(visibility, dtype=float)
    for period in continuous_periods:
        start_idx = period["start_index"]
        end_idx = period["end_index"]
        y_continuous[start_idx : end_idx + 1] = 1

    axes[1].plot(times_datetime, y_continuous, "r-", linewidth=1)
    axes[1].fill_between(times_datetime, 0, y_continuous, alpha=0.5, color="red")
    axes[1].set_ylabel(
        f'Continuous Periods\n(≥{params["visibility_threshold_hours"]}h)'
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)

    # Add period labels
    for i, period in enumerate(continuous_periods[:10]):  # Label first 10 periods
        mid_time = (
            period["start_time"] + (period["end_time"] - period["start_time"]) / 2
        )
        axes[1].annotate(
            f'{period["duration_days"]:.1f}d',
            xy=(mid_time.datetime, 0.5),
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    # Plot 3: Rolling visibility percentage (3-day window)
    window_days = 3
    window_points = int(window_days * 24 / params["time_resolution_hours"])
    if window_points > 0 and window_points < len(visibility):
        rolling_visibility = (
            np.convolve(
                visibility.astype(float),
                np.ones(window_points) / window_points,
                mode="same",
            )
            * 100
        )
        axes[2].plot(times_datetime, rolling_visibility, "g-", linewidth=1)
        axes[2].fill_between(
            times_datetime, 0, rolling_visibility, alpha=0.3, color="green"
        )
    else:
        # Fallback for short time series
        axes[2].plot(times_datetime, visibility.astype(float) * 100, "g-", linewidth=1)
        axes[2].fill_between(
            times_datetime, 0, visibility.astype(float) * 100, alpha=0.3, color="green"
        )

    axes[2].set_ylabel(f"{window_days}-Day Rolling\nVisibility %")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    # Add statistics text box
    stats_text = (
        f"Overall Visibility: {stats['visibility_percentage']:.1f}%\n"
        f"Continuous Periods: {stats['continuous_periods_count']}\n"
        f"Longest Period: {stats['longest_continuous_period_days']:.1f} days\n"
        f"Avg Period: {stats['average_continuous_period_hours']:.1f}h"
    )

    axes[0].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def plot_visibility_summary(results, figsize=(12, 8)):
    """
    Plot a summary dashboard of visibility analysis results.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    figsize : tuple
        Figure size (default: (12, 8))

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    stats = results["statistics"]
    continuous_periods = results["continuous_periods"]
    params = results["analysis_params"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f'Visibility Summary: RA {results["target_coord"].ra:.3f}°, '
        f'DEC {results["target_coord"].dec:.3f}°',
        fontsize=14,
    )

    # Plot 1: Visibility pie chart
    visible_time = stats["total_visible_hours"]
    total_time = stats["total_time_hours"]
    invisible_time = total_time - visible_time

    axes[0, 0].pie(
        [visible_time, invisible_time],
        labels=["Visible", "Not Visible"],
        colors=["lightgreen", "lightcoral"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0, 0].set_title("Overall Visibility")

    # Plot 2: Period duration histogram
    if continuous_periods:
        durations_days = [p["duration_days"] for p in continuous_periods]
        axes[0, 1].hist(
            durations_days,
            bins=min(20, len(durations_days)),
            color="skyblue",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 1].set_xlabel("Period Duration (days)")
        axes[0, 1].set_ylabel("Number of Periods")
        axes[0, 1].set_title(
            f"Continuous Period Durations\n({len(continuous_periods)} periods)"
        )
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No continuous periods\nfound",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=12,
        )
        axes[0, 1].set_title("Continuous Period Durations")

    # Plot 3: Period timeline
    if continuous_periods:
        # Show periods as horizontal bars
        y_positions = range(len(continuous_periods))
        start_times = [p["start_time"].datetime for p in continuous_periods]
        durations_hours = [p["duration_hours"] for p in continuous_periods]

        # Convert durations to matplotlib timedelta format
        durations_td = [pd.Timedelta(hours=d) for d in durations_hours]

        axes[1, 0].barh(
            y_positions,
            durations_td,
            left=start_times,
            color="orange",
            alpha=0.7,
            height=0.8,
        )
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Period Number")
        axes[1, 0].set_title("Continuous Periods Timeline")
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        # Limit to show reasonable number of periods
        if len(continuous_periods) > 20:
            axes[1, 0].set_ylim(0, 20)
            axes[1, 0].text(
                0.02,
                0.98,
                f"Showing first 20 of {len(continuous_periods)} periods",
                transform=axes[1, 0].transAxes,
                va="top",
                fontsize=8,
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No continuous periods\nto display",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Continuous Periods Timeline")

    # Plot 4: Key statistics
    axes[1, 1].axis("off")

    # Create statistics text
    stats_lines = [
        f"Analysis Period: {params['duration_days']} days",
        f"Time Resolution: {params['time_resolution_hours']} hours",
        "",
        "Overall Statistics:",
        f"• Total visible time: {stats['total_visible_hours']:.1f} h",
        f"• Visibility percentage: {stats['visibility_percentage']:.1f}%",
        "",
        f"Continuous Periods (≥{params['visibility_threshold_hours']}h):",
        f"• Number of periods: {stats['continuous_periods_count']}",
    ]

    if stats["continuous_periods_count"] > 0:
        stats_lines.extend(
            [
                f"• Longest period: {stats['longest_continuous_period_days']:.1f} days",
                f"• Average period: {stats['average_continuous_period_hours']:.1f} hours",
                f"• Total continuous time: {stats['total_continuous_hours']:.1f} h",
                f"• Continuous percentage: {stats['continuous_visibility_percentage']:.1f}%",
            ]
        )
    else:
        stats_lines.append("• No periods found")

    stats_text = "\n".join(stats_lines)
    axes[1, 1].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def plot_observation_windows(results, observation_windows, figsize=(14, 8)):
    """
    Plot optimal observation windows within continuous visibility periods.

    Parameters:
    -----------
    results : dict
        Results from analyze_yearly_visibility
    observation_windows : list
        List of observation windows from find_optimal_observation_windows
    figsize : tuple
        Figure size (default: (14, 8))

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    if not observation_windows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No observation windows found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Observation Windows")
        return fig

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    times = results["times"]
    visibility = results["visibility"]
    continuous_periods = results["continuous_periods"]

    times_datetime = [t.datetime for t in times]

    # Plot 1: Visibility with continuous periods
    axes[0].plot(times_datetime, visibility.astype(int), "b-", linewidth=0.5, alpha=0.5)
    axes[0].fill_between(
        times_datetime, 0, visibility.astype(int), alpha=0.2, color="blue"
    )

    # Highlight continuous periods
    for period in continuous_periods:
        start_dt = period["start_time"].datetime
        end_dt = period["end_time"].datetime
        axes[0].axvspan(
            start_dt,
            end_dt,
            alpha=0.3,
            color="green",
            label="Continuous Periods" if period == continuous_periods[0] else "",
        )

    axes[0].set_ylabel("Visibility")
    axes[0].set_title("Visibility Periods and Observation Windows")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Observation windows
    window_y_positions = []
    window_starts = []
    window_durations = []
    window_colors = []

    # Color map for different parent periods
    colors = plt.cm.Set3(np.linspace(0, 1, len(continuous_periods)))
    period_color_map = {
        id(period): colors[i] for i, period in enumerate(continuous_periods)
    }

    for i, window in enumerate(observation_windows):
        window_y_positions.append(i)
        window_starts.append(window["start_time"].datetime)
        window_durations.append(pd.Timedelta(hours=window["duration_hours"]))

        # Find parent period color
        parent_color = "orange"  # default
        for period in continuous_periods:
            if (
                window["start_time"] >= period["start_time"]
                and window["end_time"] <= period["end_time"]
            ):
                parent_color = period_color_map[id(period)]
                break

        window_colors.append(parent_color)

    # Plot observation windows as horizontal bars
    _ = axes[1].barh(
        window_y_positions,
        window_durations,
        left=window_starts,
        color=window_colors,
        alpha=0.8,
        height=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Observation Window #")
    axes[1].set_title(f"{len(observation_windows)} Optimal Observation Windows")
    axes[1].grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    # Add text annotation with window details
    total_obs_time = sum(window["duration_hours"] for window in observation_windows)
    info_text = (
        f"Total observation windows: {len(observation_windows)}\n"
        f"Total observation time: {total_obs_time:.1f} hours ({total_obs_time/24:.1f} days)\n"
        f"Average window duration: {total_obs_time/len(observation_windows):.1f} hours"
    )

    axes[0].text(
        0.02,
        0.02,
        info_text,
        transform=axes[0].transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig
