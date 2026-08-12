# DoomScroll Detector

An HAR (Human Activity Recognition) pipeline for joint detection of scrolling and walking from phone sensor data.

Statistical Machine Learning — Final Project, MSc in Statistical Methods and Applications, Sapienza Università di Roma (June 2026)

**Team:** Rémi Fouchérand, Dorothea Koutrintze, Mert Akyıldız, Luigi Maria Weber

> This is a fork of the team's original repository ([LuigiWeber03/Stat-Machine-Learning](https://github.com/LuigiWeber03/Stat-Machine-Learning)), kept here to host on my own profile. All code and results are joint work by the full team listed above.

## The problem

Most Human Activity Recognition research targets gross motor activity — walking, running, biking — which is well studied and routinely hits 95%+ accuracy. On-screen behavior like scrolling is a much harder signal: brief, low-amplitude, and easily confused with incidental hand motion. It's also posture-dependent — a scrolling gesture looks completely different in the sensor data depending on whether you're sitting or walking, since gait noise masks the gesture while walking.

We set out to jointly detect **posture** (sitting / walking) and **activity** (scrolling / idle) from accelerometer, gyroscope, and orientation data, and to test whether decomposing the problem — detect posture first, then detect scrolling conditioned on posture — outperforms predicting both labels at once.

## Data

- 58 sessions (2–5 min each), 2 phones, multiple locations/postures, collected via the open-source SensorLogger app at 100 Hz
- Each session has one activity × one posture, so labels are constant within a session
- Preprocessing aligns the three sensors (which fire at slightly different moments despite all targeting ~100Hz) onto a common 100Hz grid via index interpolation, then splits each session into 4 sub-sessions for variety
- 116 hand-engineered features per 4-second window (50% overlap): time-domain (std, IQR, MAD, zero-crossing rate), jerk, frequency-domain (dominant frequency, spectral entropy, band power), and structural (signal magnitude, cross-axis correlations)
- Deliberately excludes absolute features (mean, min/max) — early experiments found these leaked phone orientation, which is session-specific rather than activity-specific, and caused the model to recognize sessions instead of activities

## Approach

We tested three ways of framing the same problem, using both Random Forest and SVM+RBF:

1. **4-class joint** — one classifier predicts Sit+Idle / Sit+Scroll / Walk+Idle / Walk+Scroll directly
2. **Multi-output** — two independent binary classifiers (posture, scrolling) on the same features, trained with no shared information
3. **Cascade** (best performer) — a posture classifier first, then two separate posture-conditioned scrolling classifiers

Evaluation used `StratifiedGroupKFold` grouped by sub-session (216 groups) to prevent any leakage across folds, with macro-averaged F1 as the primary metric.

## Results

| Model / Setup | F1 macro |
|---|---|
| RF Cascade — walking detector | 0.977 ± 0.010 |
| RF Cascade — scrolling (walking context) | 0.920 ± 0.032 |
| RF Cascade — scrolling (sitting context) | 0.935 ± 0.038 |
| RF — 4-class joint | 0.913 ± 0.021 |
| SVM+RBF — 4-class joint | 0.855 ± 0.020 |
| SVM — multi-output | 0.840 ± 0.025 |

For comparison, the closest prior published result (Zhuo et al., 2020) reported 78.6% accuracy on a related 8-class problem, and still had to merge reading/scrolling labels due to confusion between them. Our cascade pushes posture-conditioned scrolling F1 to 0.92–0.94 on a harder, more granular split of the same underlying task.

**Key finding:** posture-conditioning is the bigger lever — it matters more than the Random Forest vs. SVM choice itself. Walking detection is essentially solved (F1 ≥ 0.97 everywhere); scrolling detection is where all methods diverge, and where the cascade's specialization pays off. The two posture-specific scrolling classifiers rely on genuinely different signals: the walking-context detector leans on gyroscope_z (mad/std/jerk) to separate scroll-taps from gait motion, while the sitting-context detector relies on acc_mag_mean and spectral entropy to catch a subtler, lower-energy gesture.

## Repository structure

```
preprocessing/
  merge_sessions.py       # raw JSON → common 100Hz grid → interpolation → merged session CSV
  feature_extraction.py   # 4-second windowing → 116 hand-engineered features per window
  sample_data/            # a few representative raw SensorLogger session files (see note below)
classifier.ipynb          # Random Forest: 4-class joint, multi-output, and cascade experiments
joint_svm.ipynb           # SVM+RBF: 4-class joint and multi-output experiments
```

**On the data:** the full dataset is 58 sessions collected by the team and isn't included here in full (raw sessions run several MB to 25+ MB each — several hundred MB combined, more than makes sense to version in a code repo). `preprocessing/sample_data/` has three representative raw sessions (a short test capture, one typing session, one watching session) so you can see the actual SensorLogger JSON format and run the pipeline scripts end-to-end. Ping me if you'd like access to the full 58-session set.

## Limitations

- Small subject pool — generalization to unseen users is untested
- No grid search on SVM hyperparameters (C, γ)
- Cascade is evaluated stage-wise; end-to-end error compounding across stages isn't measured (though stage-1 error is negligible: 114/4,966 windows)

## References

- Zhuo, S. et al. (2020). *Real-time smartphone activity classification using inertial sensors.* Sensors, 20(3), 655.
- Bhatele, K. R. & Bedekar, M. (2024). *Screen-gesture recognition using smartphone IMU sensors.* IJCDS.
- Saha, U. et al. (2024). *FusionActNet.* arXiv:2310.02011.
- Abdullah, M. & Ahmed, M. (2024). *Human activity recognition using statistical features and multiple classifiers.* Journal of Robotics and Control.
- Hassan, M. M. et al. (2018). *A robust human activity recognition system using smartphone sensors and deep learning.* Future Generation Computer Systems, 81, 307–313.
