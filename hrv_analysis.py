import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ==========================================
# SETTINGS
# ==========================================

fs = 25  # PPG sampling frequency in Hz


# ==========================================
# HRV PROCESSING FUNCTION
# ==========================================

def process_ppg(ppg, fs):

    # --------------------------------------
    # 1. Bandpass filter
    # --------------------------------------

    lowcut = 0.5
    highcut = 4.0

    b, a = signal.butter(
        3,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs
    )

    filtered_ppg = signal.filtfilt(
        b,
        a,
        ppg
    )

    # --------------------------------------
    # 2. Detect PPG pulse peaks
    # --------------------------------------

    min_distance = int(0.5 * fs)

    prominence = 0.30 * np.std(filtered_ppg)

    peaks, properties = signal.find_peaks(
        filtered_ppg,
        distance=min_distance,
        prominence=prominence
    )

    # --------------------------------------
    # 3. Extract beat-to-beat intervals
    # --------------------------------------

    peak_times = peaks / fs

    rr_intervals = np.diff(peak_times)

    # --------------------------------------
    # 4. Remove invalid RR intervals
    # --------------------------------------

    valid_rr = rr_intervals[
        (rr_intervals >= 0.40) &
        (rr_intervals <= 1.50)
    ]

    # Convert seconds to milliseconds

    rr_ms = valid_rr * 1000

    # --------------------------------------
    # 5. SDNN
    # --------------------------------------

    sdnn = np.std(
        rr_ms,
        ddof=1
    )

    # --------------------------------------
    # 6. RMSSD
    # --------------------------------------

    rmssd = np.sqrt(
        np.mean(
            np.diff(rr_ms) ** 2
        )
    )

    # --------------------------------------
    # 7. Interpolate RR series
    # --------------------------------------

    rr_times = np.cumsum(valid_rr)

    fs_interp = 4

    interp_time = np.arange(
        rr_times[0],
        rr_times[-1],
        1 / fs_interp
    )

    rr_interp = np.interp(
        interp_time,
        rr_times,
        rr_ms
    )

    # --------------------------------------
    # 8. Power Spectral Density
    # --------------------------------------

    frequencies, psd = signal.welch(
        rr_interp,
        fs=fs_interp,
        nperseg=min(256, len(rr_interp))
    )

    # --------------------------------------
    # 9. LF and HF power
    # --------------------------------------

    lf_mask = (
        (frequencies >= 0.04) &
        (frequencies < 0.15)
    )

    hf_mask = (
        (frequencies >= 0.15) &
        (frequencies <= 0.40)
    )

    lf_power = np.trapezoid(
        psd[lf_mask],
        frequencies[lf_mask]
    )

    hf_power = np.trapezoid(
        psd[hf_mask],
        frequencies[hf_mask]
    )

    # --------------------------------------
    # Return all results
    # --------------------------------------

    return {
        "filtered_ppg": filtered_ppg,
        "peaks": peaks,
        "rr_intervals": valid_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "frequencies": frequencies,
        "psd": psd,
        "lf_power": lf_power,
        "hf_power": hf_power
    }


# ==========================================
# PRE-INTERVENTION
# ==========================================

pre_data = pd.read_csv(
    "data/BP35_Pre/2026-02-25_12-04-39-449922_PG.csv"
)

pre_ppg = pre_data["PG"].to_numpy()

pre_results = process_ppg(
    pre_ppg,
    fs
)


# ==========================================
# DURING-INTERVENTION
# ==========================================

during_data = pd.read_csv(
    "data/BP35_During/2026-02-25_12-11-58-679694_PG.csv"
)

during_ppg = during_data["PG"].to_numpy()

# Protocol:
# 0–3 min   = Listening
# 3–7 min   = Task while listening
# 7–10 min  = Listening

during_listening_1 = during_ppg[
    0 : 3 * 60 * fs
]

during_task = during_ppg[
    3 * 60 * fs : 7 * 60 * fs
]

during_listening_2 = during_ppg[
    7 * 60 * fs : 10 * 60 * fs
]

# Complete 10-minute During period

during_ppg_10min = during_ppg[
    0 : 10 * 60 * fs
]

during_results = process_ppg(
    during_ppg_10min,
    fs
)


# ==========================================
# POST-INTERVENTION
# ==========================================

post_data = pd.read_csv(
    "data/BP35_Post/2026-02-25_12-24-51-934624_PG.csv"
)

post_ppg = post_data["PG"].to_numpy()

post_results = process_ppg(
    post_ppg,
    fs
)


# ==========================================
# PRINT CONDITION INFORMATION
# ==========================================

print("\n==========================================")
print("DATA PROCESSING SUMMARY")
print("==========================================")

print("Pre samples:", len(pre_ppg))

print("During total samples:", len(during_ppg_10min))
print("During Listening 1 samples:", len(during_listening_1))
print("During Task while listening samples:", len(during_task))
print("During Listening 2 samples:", len(during_listening_2))

print("Post samples:", len(post_ppg))


# ==========================================
# HRV COMPARISON TABLE
# ==========================================

hrv_table = pd.DataFrame({

    "Metric": [
        "SDNN (ms)",
        "RMSSD (ms)",
        "LF Power (ms²)",
        "HF Power (ms²)"
    ],

    "Pre-Intervention": [
        pre_results["sdnn"],
        pre_results["rmssd"],
        pre_results["lf_power"],
        pre_results["hf_power"]
    ],

    "During-Intervention": [
        during_results["sdnn"],
        during_results["rmssd"],
        during_results["lf_power"],
        during_results["hf_power"]
    ],

    "Post-Intervention": [
        post_results["sdnn"],
        post_results["rmssd"],
        post_results["lf_power"],
        post_results["hf_power"]
    ]
})


print("\n==========================================")
print("HRV COMPARISON TABLE")
print("==========================================")

print(
    hrv_table.round(2).to_string(index=False)
)


# ==========================================
# PERCENTAGE CHANGES
# ==========================================

pre_values = np.array([
    pre_results["sdnn"],
    pre_results["rmssd"],
    pre_results["lf_power"],
    pre_results["hf_power"]
])

during_values = np.array([
    during_results["sdnn"],
    during_results["rmssd"],
    during_results["lf_power"],
    during_results["hf_power"]
])

post_values = np.array([
    post_results["sdnn"],
    post_results["rmssd"],
    post_results["lf_power"],
    post_results["hf_power"]
])


# Pre → During

pre_to_during = (
    (during_values - pre_values)
    / pre_values
) * 100


# During → Post

during_to_post = (
    (post_values - during_values)
    / during_values
) * 100


percentage_table = pd.DataFrame({

    "Metric": [
        "SDNN (ms)",
        "RMSSD (ms)",
        "LF Power (ms²)",
        "HF Power (ms²)"
    ],

    "Pre → During (%)": pre_to_during,

    "During → Post (%)": during_to_post
})


print("\n==========================================")
print("PERCENTAGE CHANGES")
print("==========================================")

print(
    percentage_table.round(2).to_string(index=False)
)


# ==========================================
# PLOT 1:
# BEAT-TO-BEAT INTERVAL VARIATION
# ==========================================

pre_rr = pre_results["rr_intervals"]
during_rr = during_results["rr_intervals"]
post_rr = post_results["rr_intervals"]

pre_rr_time = np.cumsum(pre_rr)
during_rr_time = np.cumsum(during_rr)
post_rr_time = np.cumsum(post_rr)


plt.figure(figsize=(12, 5))

plt.plot(
    pre_rr_time,
    pre_rr * 1000,
    label="Pre-Intervention"
)

plt.plot(
    during_rr_time,
    during_rr * 1000,
    label="During-Intervention"
)

plt.plot(
    post_rr_time,
    post_rr * 1000,
    label="Post-Intervention"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Beat-to-beat interval (ms)")

plt.title("Beat-to-Beat Interval Variation")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
 # ==========================================
# PLOT:
# FILTERED PPG WITH DETECTED PEAKS
# FIRST 30 SECONDS
# ==========================================

plot_duration = 30  # seconds

pre_samples = int(plot_duration * fs)
during_samples = int(plot_duration * fs)
post_samples = int(plot_duration * fs)


plt.figure(figsize=(12, 10))


# ------------------------------------------
# PRE-INTERVENTION
# ------------------------------------------

plt.subplot(3, 1, 1)

pre_signal = pre_results["filtered_ppg"][:pre_samples]

pre_time = np.arange(len(pre_signal)) / fs

pre_peaks = pre_results["peaks"][
    pre_results["peaks"] < pre_samples
]

plt.plot(
    pre_time,
    pre_signal,
    label="Filtered PPG"
)

plt.scatter(
    pre_peaks / fs,
    pre_results["filtered_ppg"][pre_peaks],
    marker="x",
    label="Detected Peaks"
)

plt.title(
    "Pre-Intervention: filtered PPG with detected peaks (first 30 s)"
)

plt.xlabel("Time (s)")
plt.ylabel("Filtered PPG")

plt.legend()
plt.grid(True)


# ------------------------------------------
# DURING-INTERVENTION
# ------------------------------------------

plt.subplot(3, 1, 2)

during_signal = during_results["filtered_ppg"][:during_samples]

during_time = np.arange(len(during_signal)) / fs

during_peaks = during_results["peaks"][
    during_results["peaks"] < during_samples
]

plt.plot(
    during_time,
    during_signal,
    label="Filtered PPG"
)

plt.scatter(
    during_peaks / fs,
    during_results["filtered_ppg"][during_peaks],
    marker="x",
    label="Detected Peaks"
)

plt.title(
    "During Intervention: filtered PPG with detected peaks (first 30 s)"
)

plt.xlabel("Time (s)")
plt.ylabel("Filtered PPG")

plt.legend()
plt.grid(True)


# ------------------------------------------
# POST-INTERVENTION
# ------------------------------------------

plt.subplot(3, 1, 3)

post_signal = post_results["filtered_ppg"][:post_samples]

post_time = np.arange(len(post_signal)) / fs

post_peaks = post_results["peaks"][
    post_results["peaks"] < post_samples
]

plt.plot(
    post_time,
    post_signal,
    label="Filtered PPG"
)

plt.scatter(
    post_peaks / fs,
    post_results["filtered_ppg"][post_peaks],
    marker="x",
    label="Detected Peaks"
)

plt.title(
    "Post-Intervention: filtered PPG with detected peaks (first 30 s)"
)

plt.xlabel("Time (s)")
plt.ylabel("Filtered PPG")

plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()


# ==========================================
# PLOT 2:
# SDNN AND RMSSD COMPARISON
# ==========================================

conditions = [
    "Pre-Intervention",
    "During-Intervention",
    "Post-Intervention"
]

sdnn_values = [
    pre_results["sdnn"],
    during_results["sdnn"],
    post_results["sdnn"]
]

rmssd_values = [
    pre_results["rmssd"],
    during_results["rmssd"],
    post_results["rmssd"]
]

x = np.arange(len(conditions))

width = 0.35


plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    sdnn_values,
    width,
    label="SDNN"
)

plt.bar(
    x + width / 2,
    rmssd_values,
    width,
    label="RMSSD"
)

plt.xlabel("Condition")
plt.ylabel("HRV (ms)")

plt.title("SDNN and RMSSD Comparison")

plt.xticks(
    x,
    conditions
)

plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.show()


# ==========================================
# PLOT 3:
# PSD COMPARISON
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    pre_results["frequencies"],
    pre_results["psd"],
    label="Pre-Intervention"
)

plt.plot(
    during_results["frequencies"],
    during_results["psd"],
    label="During-Intervention"
)

plt.plot(
    post_results["frequencies"],
    post_results["psd"],
    label="Post-Intervention"
)


# LF/HF boundaries

plt.axvline(
    0.04,
    linestyle="--",
    label="LF lower limit"
)

plt.axvline(
    0.15,
    linestyle="--",
    label="LF/HF boundary"
)

plt.axvline(
    0.40,
    linestyle="--",
    label="HF upper limit"
)

plt.xlim(0, 0.5)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (ms²/Hz)")

plt.title("HRV Power Spectral Density Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# PLOT 4:
# LF AND HF POWER COMPARISON
# ==========================================

lf_values = [
    pre_results["lf_power"],
    during_results["lf_power"],
    post_results["lf_power"]
]

hf_values = [
    pre_results["hf_power"],
    during_results["hf_power"],
    post_results["hf_power"]
]


plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    lf_values,
    width,
    label="LF Power"
)

plt.bar(
    x + width / 2,
    hf_values,
    width,
    label="HF Power"
)

plt.xlabel("Condition")
plt.ylabel("Power (ms²)")

plt.title("LF and HF Power Comparison")

plt.xticks(
    x,
    conditions
)

plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.show()