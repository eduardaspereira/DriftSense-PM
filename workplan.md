                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026




DriftSense-PM – Detailed Technical Plan with Experimental Design

Scientific Goal
   •     Design controlled drift-aware predictive maintenance benchmark.
   •     Evaluate detection and adaptation mechanisms.

Fully 15-Week Plan

Week 1 – Sensor Setup, Calibration & Signal Validation

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating training data, or reporting only averaged results without raw logs.

   • Assemble PM kit.
   • Validate sensor calibration.
   • Implement signal logging.
The drifts will use different scenarios(D0–D5) as detailed below:
   • D0 – No Drift (Control Scenario) - This validates detector specificity.
           o Baseline dataset.
           o Used to measure:
                     False positives of detectors
                     Unnecessary retraining cost
                     Stability
   • D1 – Temperature Drift (Covariate Drift)
           o What changes?
                     Ambient temperature increases/decreases.
                     Affects vibration amplitude or frequency distribution.
           o Type: Feature distribution shift
           o Realistic case: Factory seasonal variation.
           o How to create it physically:
                     Heat gun (controlled distance).
                     Thermal chamber.
                     Extended runtime heating.
                     Log temperature continuously.
   • D2 – Mounting Drift (Mechanical Drift)
           o What changes?
                     Sensor position/orientation changes.
                     Alters vibration signal characteristics.



       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                MEI, 1st year, Project in Internet Engineering, 2025/2026

            o    Type: Sensor installation shift
            o    Realistic case: Sensor reinstallation or maintenance.
            o    How to create it:
                      Slightly loosen mounting screw.
                      Change orientation angle.
                      Add damping material under sensor.
            o Record angle/position in DATASET.md.
   • D3 – Regime Drift (Operational Drift)
            o What changes?
                      Machine operating speed/load changes.
                      Type: Regime shift
            o Realistic case: Different production load.
            o How to create it:
                      Change motor RPM.
                      Change load weight.
                      Change duty cycle.
            o This shifts frequency-domain features significantly.
   • D4 – Bias / Noise Drift (Sensor Degradation)
            o What changes?
                      Add noise or offset to signal.
                      Type: Noise injection / bias shift
            o Realistic case: Sensor aging.
            o How to create it:
                      Add artificial Gaussian noise in preprocessing.
                      Add constant bias offset.
                      Use external EMI source.
   • D5 – Combined Drift (Realistic Scenario)
            o Combination of:
                      Temperature + Mounting
                      Regime + Noise
                      Or all combined
            o This is your most realistic industrial scenario.
            o Used to evaluate robustness under compound degradation.
The drifts will use different scenarios(D0–D5) as detailed below:
   • Det0 – None (Baseline)
            o No drift detection.
            o Used to measure:
                      Natural degradation curve
                      Cost of not detecting drift
   • Det1 – Error Monitoring (Performance-Based Detector)
            o Monitors:
                      F1-score degradation
                      Prediction error trend



     © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026

                     Confidence score drift
            o Triggers if:
                     Error exceeds threshold
                     Sustained degradation over N windows
            o Type: Concept drift detector
            o Pros:
                     Directly tied to task performance
            o Cons:
                     Needs labeled data or proxy signal
   •     Det2 – Distribution Test (Statistical Detector)
            o Monitors:
                     Feature distribution shift
                     Kullback–Leibler (KL) divergence
                     Kolmogorov–Smirnov (KS) test
                     PSI (Population Stability Index)
                     Use SciPy (https://scipy.org/ ) for these tests
            o Type: Covariate drift detector
            o Pros:
                     Does not need labels
                     Earlier detection possible
            o Cons:
                     May trigger false positives

The different Adaptation Strategies (A0–A2) are:

   •     A0 – None
            o No adaptation.
            o Used to measure:
                     Raw degradation
                     Recovery baseline (none)
   •     A1 – Periodic Retraining
            o Retrain every fixed interval:
                     Every X minutes
                     Every Y windows
            o Independent of detection.
            o Measures:
                     Energy cost
                     Latency
                     Recovery time
   •     A2 – Lightweight Adaptation
            o Triggered retraining but:
                     Small incremental update
                     Fine-tuning only
                     Feature recalibration



       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

                         Model parameter shift
                         Threshold recalibration
               o     Low energy footprint.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      Signal stability variance.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Validated raw dataset sample.

Week 2 – Data Acquisition Pipeline & Structured Logging Design

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating training data, or reporting only averaged results without raw logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 3 – Controlled Drift Taxonomy & Experimental Protocol Specification

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating training data, or reporting only averaged results without raw logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 4 – Baseline Dataset Collection & Integrity Verification
  • Baseline dataset complete: ≥1000 windows per state (normal/slight/strong) with
       balanced classes.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

    •     Drift protocol documented: D1–D5 procedures reproducible and recorded in
          DATASET.md.
    •     Integrity checks passed: sampling rate, windowing, timestamps, and labels
          validated.
    •     Go/No-Go: proceed only if dataset v1.0 is frozen and backed up.

Milestone Gate Review (Week 4 Checkpoint)

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating training data, or reporting only averaged results without raw logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 5 – Feature Engineering Pipeline (Time & Frequency Domain)

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 6 – Baseline Predictive Maintenance Model Training & Validation

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating training data, or reporting only averaged results without raw logs.

   •      Execute dataset/model/detector tasks.
   •      Run controlled drift experiments.
   •      Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

    •     F1-score.
    •     Detection delay.
    •     Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 7 – Single-Drift Scenario Injection & Performance Degradation Analysis

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 8 – Combined Drift Construction & Degradation Curve Modeling
  • Baseline model frozen: static PM model trained and evaluated on baseline dataset.
  • Drift datasets complete: D1–D3 + D5 collected; degradation curves produced.
  • Decision: lock detector thresholds plan and finalize drift evaluation protocol.

Milestone Gate Review (Week 8 Checkpoint)

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Week 9 – Drift Detection Algorithm Implementation & Threshold Calibration

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 10 – Drift Detector Evaluation (Delay, FPR & Robustness)

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

   •      Execute dataset/model/detector tasks.



        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 11 – Periodic Retraining Strategy Implementation & Cost Analysis

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

   •      Execute dataset/model/detector tasks.
   •      Run controlled drift experiments.
   •      Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.



        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

    •     Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 12 – Lightweight Adaptation Strategy & Comparative Recovery Analysis
  • Detectors evaluated: detection delay and false-positive rates measured across drifts.
  • Adaptation implemented: periodic retrain and lightweight adaptation both working
      end-to-end.
  • Decision: finalize full factorial evaluation scripts and begin final campaign runs.

Milestone Gate Review (Week 12 Checkpoint)

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 13 – Automated Full-Factorial Evaluation Campaign

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

    •     F1-score.
    •     Detection delay.
    •     Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Week 14 – Statistical Analysis, Confidence Intervals & Significance Testing

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

    •     Execute dataset/model/detector tasks.
    •     Run controlled drift experiments.
    •     Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

   •      F1-score.
   •      Detection delay.
   •      Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

   •      Updated plots and logs.

Week 15 – Reproducibility Validation, Threats to Validity & Paper Finalization

Technical Tasks
In this section, you must implement the ML pipeline components for the week. Every subtask
must produce executable code, versioned datasets or models, and documented
hyperparameters. Reproducibility and controlled comparisons are mandatory.

Common mistakes to avoid: modifying datasets between experiments, tuning without
documentation, evaluating on training data, or reporting only averaged results without raw
logs.

   •      Execute dataset/model/detector tasks.
   •      Run controlled drift experiments.
   •      Document parameter settings.

Evaluation Metrics
Here you must compute ML performance and robustness metrics programmatically (F1-score,
detection delay, FPR, recovery time). All results must come from scripts and stored raw logs.




        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                   MEI, 1st year, Project in Internet Engineering, 2025/2026

Common mistakes to avoid: reporting only accuracy, ignoring imbalance, skipping statistical
tests, or copying values manually.

    •     F1-score.
    •     Detection delay.
    •     Recovery time.

Deliverables
Deliverables must include trained models, detectors, adaptation scripts, structured logs, and
publication-ready plots. Everything must be reproducible from the repository.

Common mistakes to avoid: submitting plots without raw data, not versioning models, or
producing non-reproducible results.

    •     Updated plots and logs.

Full Factorial Experiment Design Matrix
 Drift Scenario              Detector                      Adaptation                            Device                 Rep.
                             Det0: None                    A0: None                              Pi (training) +            5
 D0: No drift                Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
                             Det0: None                    A0: None                              Pi (training) +            5
 D1: Temperature             Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
                             Det0: None                    A0: None                              Pi (training) +            5
 D2: Mounting                Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
                             Det0: None                    A0: None                              Pi (training) +            5
 D3: Regime                  Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
                             Det0: None                    A0: None                              Pi (training) +            5
 D4: Bias/Noise              Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
                             Det0: None                    A0: None                              Pi (training) +
 D5: Combined                Det1: Error Monitoring        A1: Periodic Retrain                  MCU (optional
                             Det2: Distribution Test       A2: Lightweight Adapt                 inference)
Execution (fixed for all runs)

          •     Pi (training / detection / adaptation) + MCU (optional inference)

Repetitions (fixed for all runs)

    •     5 repeats per configuration

Statistical Validation Plan
    •     5–5 repetitions per scenario.
    •     Mean ± std reporting.
    •     Wilcoxon signed-rank test for detector comparison.
    •     Confidence intervals (95%).

Evaluation Workflow Diagram
Data Collection → Data Preprocessing → Feature Extraction → Model Training



        © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026

→ Baseline Evaluation → Drift/Shift Injection → Detection/Policy Execution

→ Adaptation/Offloading Decision → Metric Logging → Statistical Analysis

Required Hardware
   •     Arduino Pro Smart Industry Predictive Maintenance Kit (different sensors; stream
         data to the RPi 5)
   •     Raspberry Pi 5 (Feature extraction; Drift detection; Model training; Logging;
         Statistical evaluation)
   •     USB power meter
   •     Mechanical drift setup, in this case, the DC motor (using the Smart Fan setup but
         controlled by the Raspberry Pi 5)
   •     External storage

Exhaustive Factorial Experiment Matrix (Explicit Combinations)
Drift Scenario          Detector                  Adaptation                Execution                 Repetitions
D0 (No drift)           Det0 (None)               A0 (None)                 Pi (training) +           5
                                                                            MCU (optional
                                                                            inference)
D0 (No drift)           Det0 (None)               A1 (Periodic              Pi (training) +           5
                                                  Retrain)                  MCU (optional
                                                                            inference)
D0 (No drift)           Det0 (None)               A2                        Pi (training) +           5
                                                  (Lightweight              MCU (optional
                                                  Adapt)                    inference)
D0 (No drift)           Det1 (Error               A0 (None)                 Pi (training) +           5
                        Monitoring)                                         MCU (optional
                                                                            inference)
D0 (No drift)           Det1 (Error               A1 (Periodic              Pi (training) +           5
                        Monitoring)               Retrain)                  MCU (optional
                                                                            inference)
D0 (No drift)           Det1 (Error               A2                        Pi (training) +           5
                        Monitoring)               (Lightweight              MCU (optional
                                                  Adapt)                    inference)
D0 (No drift)           Det2                      A0 (None)                 Pi (training) +           5
                        (Distribution                                       MCU (optional
                        Test)                                               inference)
D0 (No drift)           Det2                      A1 (Periodic              Pi (training) +           5
                        (Distribution             Retrain)                  MCU (optional
                        Test)                                               inference)
D0 (No drift)           Det2                      A2                        Pi (training) +           5
                        (Distribution             (Lightweight              MCU (optional
                        Test)                     Adapt)                    inference)
D1                      Det0 (None)               A0 (None)                 Pi (training) +           5
(Temperature)                                                               MCU (optional
                                                                            inference)
D1                      Det0 (None)               A1 (Periodic              Pi (training) +           5
(Temperature)                                     Retrain)                  MCU (optional



       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                MEI, 1st year, Project in Internet Engineering, 2025/2026

                                                                          inference)
D1                    Det0 (None)               A2                        Pi (training) +           5
(Temperature)                                   (Lightweight              MCU (optional
                                                Adapt)                    inference)
D1                    Det1 (Error               A0 (None)                 Pi (training) +           5
(Temperature)         Monitoring)                                         MCU (optional
                                                                          inference)
D1                    Det1 (Error               A1 (Periodic              Pi (training) +           5
(Temperature)         Monitoring)               Retrain)                  MCU (optional
                                                                          inference)
D1                    Det1 (Error               A2                        Pi (training) +           5
(Temperature)         Monitoring)               (Lightweight              MCU (optional
                                                Adapt)                    inference)
D1                    Det2                      A0 (None)                 Pi (training) +           5
(Temperature)         (Distribution                                       MCU (optional
                      Test)                                               inference)
D1                    Det2                      A1 (Periodic              Pi (training) +           5
(Temperature)         (Distribution             Retrain)                  MCU (optional
                      Test)                                               inference)
D1                    Det2                      A2                        Pi (training) +           5
(Temperature)         (Distribution             (Lightweight              MCU (optional
                      Test)                     Adapt)                    inference)
D2 (Mounting)         Det0 (None)               A0 (None)                 Pi (training) +           5
                                                                          MCU (optional
                                                                          inference)
D2 (Mounting)         Det0 (None)               A1 (Periodic              Pi (training) +           5
                                                Retrain)                  MCU (optional
                                                                          inference)
D2 (Mounting)         Det0 (None)               A2                        Pi (training) +           5
                                                (Lightweight              MCU (optional
                                                Adapt)                    inference)
D2 (Mounting)         Det1 (Error               A0 (None)                 Pi (training) +           5
                      Monitoring)                                         MCU (optional
                                                                          inference)
D2 (Mounting)         Det1 (Error               A1 (Periodic              Pi (training) +           5
                      Monitoring)               Retrain)                  MCU (optional
                                                                          inference)
D2 (Mounting)         Det1 (Error               A2                        Pi (training) +           5
                      Monitoring)               (Lightweight              MCU (optional
                                                Adapt)                    inference)
D2 (Mounting)         Det2                      A0 (None)                 Pi (training) +           5
                      (Distribution                                       MCU (optional
                      Test)                                               inference)
D2 (Mounting)         Det2                      A1 (Periodic              Pi (training) +           5
                      (Distribution             Retrain)                  MCU (optional
                      Test)                                               inference)
D2 (Mounting)         Det2                      A2                        Pi (training) +           5
                      (Distribution             (Lightweight              MCU (optional
                      Test)                     Adapt)                    inference)



     © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                MEI, 1st year, Project in Internet Engineering, 2025/2026

D3 (Regime)           Det0 (None)               A0 (None)                 Pi (training) +           5
                                                                          MCU (optional
                                                                          inference)
D3 (Regime)           Det0 (None)               A1 (Periodic              Pi (training) +           5
                                                Retrain)                  MCU (optional
                                                                          inference)
D3 (Regime)           Det0 (None)               A2                        Pi (training) +           5
                                                (Lightweight              MCU (optional
                                                Adapt)                    inference)
D3 (Regime)           Det1 (Error               A0 (None)                 Pi (training) +           5
                      Monitoring)                                         MCU (optional
                                                                          inference)
D3 (Regime)           Det1 (Error               A1 (Periodic              Pi (training) +           5
                      Monitoring)               Retrain)                  MCU (optional
                                                                          inference)
D3 (Regime)           Det1 (Error               A2                        Pi (training) +           5
                      Monitoring)               (Lightweight              MCU (optional
                                                Adapt)                    inference)
D3 (Regime)           Det2                      A0 (None)                 Pi (training) +           5
                      (Distribution                                       MCU (optional
                      Test)                                               inference)
D3 (Regime)           Det2                      A1 (Periodic              Pi (training) +           5
                      (Distribution             Retrain)                  MCU (optional
                      Test)                                               inference)
D3 (Regime)           Det2                      A2                        Pi (training) +           5
                      (Distribution             (Lightweight              MCU (optional
                      Test)                     Adapt)                    inference)
D4                    Det0 (None)               A0 (None)                 Pi (training) +           5
(Bias/Noise)                                                              MCU (optional
                                                                          inference)
D4                    Det0 (None)               A1 (Periodic              Pi (training) +           5
(Bias/Noise)                                    Retrain)                  MCU (optional
                                                                          inference)
D4                    Det0 (None)               A2                        Pi (training) +           5
(Bias/Noise)                                    (Lightweight              MCU (optional
                                                Adapt)                    inference)
D4                    Det1 (Error               A0 (None)                 Pi (training) +           5
(Bias/Noise)          Monitoring)                                         MCU (optional
                                                                          inference)
D4                    Det1 (Error               A1 (Periodic              Pi (training) +           5
(Bias/Noise)          Monitoring)               Retrain)                  MCU (optional
                                                                          inference)
D4                    Det1 (Error               A2                        Pi (training) +           5
(Bias/Noise)          Monitoring)               (Lightweight              MCU (optional
                                                Adapt)                    inference)
D4                    Det2                      A0 (None)                 Pi (training) +           5
(Bias/Noise)          (Distribution                                       MCU (optional
                      Test)                                               inference)
D4                    Det2                      A1 (Periodic              Pi (training) +           5



     © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026

(Bias/Noise)            (Distribution             Retrain)                  MCU (optional
                        Test)                                               inference)
D4                      Det2                      A2                        Pi (training) +           5
(Bias/Noise)            (Distribution             (Lightweight              MCU (optional
                        Test)                     Adapt)                    inference)
D5 (Combined)           Det0 (None)               A0 (None)                 Pi (training) +           5
                                                                            MCU (optional
                                                                            inference)
D5 (Combined)           Det0 (None)               A1 (Periodic              Pi (training) +           5
                                                  Retrain)                  MCU (optional
                                                                            inference)
D5 (Combined)           Det0 (None)               A2                        Pi (training) +           5
                                                  (Lightweight              MCU (optional
                                                  Adapt)                    inference)
D5 (Combined)           Det1 (Error               A0 (None)                 Pi (training) +           5
                        Monitoring)                                         MCU (optional
                                                                            inference)
D5 (Combined)           Det1 (Error               A1 (Periodic              Pi (training) +           5
                        Monitoring)               Retrain)                  MCU (optional
                                                                            inference)
D5 (Combined)           Det1 (Error               A2                        Pi (training) +           5
                        Monitoring)               (Lightweight              MCU (optional
                                                  Adapt)                    inference)
D5 (Combined)           Det2                      A0 (None)                 Pi (training) +           5
                        (Distribution                                       MCU (optional
                        Test)                                               inference)
D5 (Combined)           Det2                      A1 (Periodic              Pi (training) +           5
                        (Distribution             Retrain)                  MCU (optional
                        Test)                                               inference)
D5 (Combined)           Det2                      A2                        Pi (training) +           5
                        (Distribution             (Lightweight              MCU (optional
                        Test)                     Adapt)                    inference)


Reproducibility & Artifact Packaging (Top-Conference Ready)
   •     Repository structure must separate: (i) data, (ii) code, (iii) configs, (iv) results, and
         (v) paper.
   •     All experiments must run from a single entrypoint script (e.g., run_experiment.py)
         with a YAML/JSON config.
   •     Every run must log: run_id, git_commit, config hash, random seed, device identifiers,
         timestamps, and metrics.
   •     Pin dependencies (requirements.txt) and provide an environment lock (conda
         env.yml or poetry.lock).
   •     Provide a Dockerfile (recommended) to reproduce the analysis pipeline on any
         machine.
   •     Dataset versioning: freeze a 'v1.0' dataset release and never overwrite; add a
         DATASET.md describing collection protocol and labeling.



       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026

   •     Include a REPRODUCIBILITY.md with: hardware setup photos/diagrams, exact
         placement constraints, and step-by-step reproduction commands.
   •     Artifact contents checklist (minimum): source code, configs, small sample dataset
         (or traces), scripts to regenerate all plots/tables, and a results CSV bundle.
   •     Statistical reproducibility: store raw per-run metrics (not only averages) and report
         confidence intervals with scripts.
   •     Packaging for submission: create an 'artifact.zip' and, if allowed, publish to Zenodo
         to obtain a DOI (after acceptance).
   •     Recommended directory layout:

DriftSense-PM/
 README.md
 REPRODUCIBILITY.md
 DATASET.md
 LICENSE
 env/
  requirements.txt
  environment.yml
  Dockerfile
 configs/
 src/
 scripts/
 data/
  raw/
  processed/
  splits/
 results/
  logs/
  metrics/
  figures/
 paper/
  overleaf/


Formal Milestone Gate Review – GO / NO-GO Criteria

Week 4 – Dataset Validation Gate
  • GO if: Required minimum samples collected; class imbalance < 10%; metadata
       complete; dataset version frozen (v1.0).
  • GO if: Single-command reproducibility script generates dataset summary statistics.
  • NO-GO if: Missing labels, corrupted samples, undocumented preprocessing steps, or
       non-reproducible splits.




       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
                                                  MEI, 1st year, Project in Internet Engineering, 2025/2026

Week 8 – Baseline & Robustness Validation Gate
  • GO if: Baseline models frozen; degradation curves generated; shift/drift protocols
       validated and documented.
  • GO if: Latency (p95) stable within ±10% across repetitions.
  • NO-GO if: Metrics inconsistent across runs; pipeline unstable; undocumented
       parameter tuning.

Week 12 – Experimental Readiness Gate
  • GO if: All policies/detectors operational; energy & latency logging validated;
      factorial script executes end-to-end.
  • GO if: Minimum 5 repetitions successfully logged for test configurations.
  • NO-GO if: Automation failures; inconsistent energy readings; reproducibility not
      verified.

ACM Artifact Evaluation & Badging Plan (ACM-Style)
   •     Target ACM Artifact Review Badges: Artifacts Evaluated – Functional, Artifacts
         Evaluated – Reusable.
   •     Provide public repository with tagged release matching paper submission version.
   •     Include INSTALL.md with complete hardware and software setup instructions.
   •     Include RUN.md with exact commands to reproduce every main figure and table.
   •     Provide Docker container or environment.yml for full dependency control.
   •     Package reduced validation dataset (<500MB) for artifact reviewers.
   •     Provide full raw logs and scripts that regenerate plots automatically.
   •     Include detailed hardware setup diagrams and calibration procedures.
   •     Specify runtime expectations and compute requirements.
   •     Prepare Artifact Appendix in paper describing verification workflow and expected
         outputs.




       © Prof. Flávio de Oliveira Silva, Ph.D. – flavio@di.uminho.pt - For academic use only. Redistribution prohibited.
