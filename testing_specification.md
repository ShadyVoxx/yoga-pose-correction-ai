# Testing Specification: Yoga Posture Detection & Correction

This document outlines the input, process, and output for the testing phases of the Yoga Posture Detection system. The testing is broken down into two main modules.

## Module 1: Testing the Machine Learning Model (Pose Detection & Spatiotemporal Analysis)

**Input:** Raw sequence of video frames from the webcam, containing 3D skeletal landmark coordinates (x, y, z for 33 joints) extracted via Mediapipe, and sequential frame history for temporal reasoning.

The ML testing module is responsible for evaluating the trained model’s ability to accurately and consistently identify the user's high-level yoga poses in real time. The process flow of the machine learning testing phase involves passing unseen validation data through the trained network. A shared Spatiotemporal Recurrent Neural Network (RNN/LSTM) processes sequential landmark observations to capture temporal dependencies, ensuring that normal movement variations do not cause jitter or "pose-sticking." During this testing phase, weights are frozen, and the model evaluates the state values purely on global frame information.

The neural architecture consists of a spatial feature extraction layer (calculating relative limb distances and angles), a recurrent core (e.g., LSTM hidden units) for temporal reasoning, and a classification head outputting logits for the discrete supported yoga poses. 

Testing follows a standard validation framework, measuring metrics such as Precision, Recall, and F1-score across thousands of test images to ensure the model generalizes well to new users of varying body types and environments. 

**Output:** Discrete yoga pose classifications (e.g., "Downward Dog", "Warrior II"), continuous confidence scores for the current posture, and smoothed, jitter-free skeletal coordinates for temporal continuity.

*Figure 4.1 shows the ML evaluation matrix at Step 15, where the recurrent model successfully classifies active transitions between poses without dropping frames or misclassifying intermediate movements.*

***

## Module 2: Testing the Posture Correction Engine (Heuristics & Feedback Generation)

**Input:** The classified pose label from Module 1, the user's real-time dynamic joint angles, and the statistically-derived heuristic dataset containing the "ideal" baseline angles for the target posture.

The correction testing module is responsible for learning and generating granular, actionable feedback by comparing the user to structural ideals. The process flow of the correction module evaluates the user's localized geometry. A thresholding algorithm calculates the mathematical delta between the user's live joint angles and the dataset's statistical baseline. 

The evaluation architecture acts as a deterministic, rule-based critic. It isolates specific kinetic chains (e.g., the angle between the hip, knee, and ankle) and applies a visibility and tolerance threshold. If the calculated deviation exceeds strict algorithmic boundaries, the sub-routine triggers a localized error flag. Testing this module involves feeding it synthetic "incorrect" posture data (e.g., a bent knee in a pose that requires a straight leg) to ensure that the system consistently issues the correct geometry adjustment without generating false positives.

The pseudocode for the Heuristic Comparison and Correction generation calculates the absolute difference matrix between the live tensor and the baseline matrix, mapping results to natural language outputs.

**Output:** Granular, attention-based posture corrections (e.g., "Straighten your left knee by 15 degrees", "Lower your hips"), an overall pose accuracy percentage, and mapped coordinate instructions.

***

## Appendix: Step-by-Step Computational Test Cases

### Test Case 1: Downward Dog (Adho Mukha Svanasana) - Leg Straightness

**Objective:** Validate that the system correctly identifies bent knees in Downward Dog and calculates the required angular adjustment.

**Input Data (Synthetic Mediapipe Coordinates in 2D space for simplicity):**
*   **Target Kinetic Chain:** Left Hip (Joint A) -> Left Knee (Joint B) -> Left Ankle (Joint C)
*   **Coordinates:**
    *   Left Hip (A): `(x: 10, y: 50)`
    *   Left Knee (B): `(x: 20, y: 25)`
    *   Left Ankle (C): `(x: 35, y: 5)`

**Step-by-Step Computations:**

**Step 1: Vector Creation**
Calculate spatial vectors radiating from the vertex joint (Knee B).
*   Vector `BA` (Knee to Hip) = `[A.x - B.x, A.y - B.y]` = `[10 - 20, 50 - 25]` = `[-10, 25]`
*   Vector `BC` (Knee to Ankle) = `[C.x - B.x, C.y - B.y]` = `[35 - 20, 5 - 25]` = `[15, -20]`

**Step 2: Magnitude and Dot Product**
*   Magnitude `||BA||` = `sqrt((-10)² + 25²)` = `sqrt(100 + 625)` = `sqrt(725)` ≈ `26.9`
*   Magnitude `||BC||` = `sqrt(15² + (-20)²)` = `sqrt(225 + 400)` = `sqrt(625)` = `25.0`
*   Dot Product `(BA · BC)` = `(-10 * 15) + (25 * -20)` = `-150 - 500` = `-650`

**Step 3: Angle Calculation (Cosine Theorem)**
*   `cos(θ)` = `(BA · BC) / (||BA|| * ||BC||)`
*   `cos(θ)` = `-650 / (26.9 * 25.0)` = `-650 / 672.5` ≈ `-0.966`
*   `Angle (θ) in Radians` = `arccos(-0.966)` ≈ `2.87`
*   `Angle (θ) in Degrees` = `2.87 * (180/π)` ≈ `164.5°`

**Step 4: Heuristic Comparison and Generating Feedback**
*   **Statistical Baseline (from Dataset):** Mean Knee Angle for Downward Dog = `175°`
*   **Tolerance Threshold:** `±10°` (Acceptable range is `165° - 180°`)
*   **Evaluation:** The measured angle (`164.5°`) is fractionally below the acceptable threshold.
*   **Mathematical Delta:** `175° - 164.5°` = `10.5°` deviation.
*   **Output Trigger:** *Error Event Logged.*
*   **Final Correction Output:** "Straighten your left knee by approximately 11 degrees."

---

### Test Case 2: Warrior II (Virabhadrasana II) - Front Knee Bend

**Objective:** Validate that the system identifies if the user's front lunge is too shallow and computes the required angle.

**Input Data (Synthetic Mediapipe Coordinates):**
*   **Target Kinetic Chain:** Right Hip (Joint A) -> Right Knee (Joint B) -> Right Ankle (Joint C)
*   *Note: In a proper Warrior II, the front thigh should ideally be parallel to the floor, creating a 90° angle at the knee.*
*   **Coordinates:**
    *   Right Hip (A): `(x: 40, y: 50)`
    *   Right Knee (B): `(x: 70, y: 50)`
    *   Right Ankle (C): `(x: 80, y: 15)`

**Step-by-Step Computations:**

**Step 1: Vector Creation**
*   Vector `BA` (Knee to Hip) = `[40 - 70, 50 - 50]` = `[-30, 0]`
*   Vector `BC` (Knee to Ankle) = `[80 - 70, 15 - 50]` = `[10, -35]`

**Step 2: Magnitude and Dot Product**
*   Magnitude `||BA||` = `sqrt((-30)² + 0²)` = `30.0`
*   Magnitude `||BC||` = `sqrt(10² + (-35)²)` = `sqrt(100 + 1225)` = `sqrt(1325)` ≈ `36.4`
*   Dot Product `(BA · BC)` = `(-30 * 10) + (0 * -35)` = `-300 + 0` = `-300`

**Step 3: Angle Calculation**
*   `cos(θ)` = `-300 / (30.0 * 36.4)` = `-300 / 1092` ≈ `-0.274`
*   `Angle (θ) in Radians` = `arccos(-0.274)` ≈ `1.848`
*   `Angle (θ) in Degrees` = `1.848 * (180/π)` ≈ `105.9°`

**Step 4: Heuristic Comparison and Generating Feedback**
*   **Statistical Baseline (from Dataset):** Mean Knee Angle for Warrior II (Front Leg) = `90°`
*   **Tolerance Threshold:** `±15°` (Acceptable range is `75° - 105°`)
*   **Evaluation:** The measured angle (`105.9°`) exceeds the maximum tolerance, meaning the leg is too straight.
*   **Mathematical Delta:** `105.9° - 90°` = `15.9°` deviation.
*   **Output Trigger:** *Error Event Logged.*
*   **Final Correction Output:** "Bend your front knee lower by 16 degrees."
