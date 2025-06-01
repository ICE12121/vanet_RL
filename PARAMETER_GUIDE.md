# 📖 Complete Parameter Guide: Handover Score & Negative Effects

## 🎯 Overview

This guide provides a comprehensive explanation of all parameters used in the VANET handover decision making system, including the mathematical formulas, real-world meanings, and code implementations.

## 🏆 **Core Handover Score Formula**

```python
Handover Score = α₁ × Signal_Strength + α₂ × Load_Factor + α₃ × Time_to_Stay - α₄ × Total_Penalty
```

**Location in Code**: `vanet_env/environment.py`, lines 369-374

```python
score = (
    self.alpha1 * signal_strength +     # 0.4 × Signal component
    self.alpha2 * load_factor +         # 0.3 × Load balancing component  
    self.alpha3 * time_to_stay -        # 0.2 × Stability component
    self.alpha4 * total_penalty         # 0.1 × Penalty component
)
```

---

## ⚖️ **Weight Parameters (α₁, α₂, α₃, α₄)**

### **α₁ = 0.4 (Signal Strength Weight)**

**Code Location**: `vanet_env/environment.py`, line 57
```python
self.alpha1 = 0.4  # Signal strength weight
```

**Purpose**: Controls how much importance is given to signal quality vs other factors

**Real-world Impact**:
- **Higher values** (0.6-0.8): Prioritize strong connections, good for high-bandwidth applications
- **Lower values** (0.2-0.3): Allow weaker connections if other factors are good
- **Our choice (0.4)**: Balanced approach giving moderate priority to signal quality

**Mathematical Effect**: Every 10 points of signal strength contributes 4 points to final score

---

### **α₂ = 0.3 (Load Factor Weight)**

**Code Location**: `vanet_env/environment.py`, line 58
```python
self.alpha2 = 0.3  # RSU load weight
```

**Purpose**: Controls how much the system prioritizes load balancing

**Formula Applied To**:
```python
load_factor = 1 / (rsu['current_load'] / rsu['capacity'] + 1e-6)
```

**Real-world Impact**:
- **Higher values** (0.4-0.6): Strong load balancing, prevents RSU overload
- **Lower values** (0.1-0.2): Less concern about RSU congestion
- **Our choice (0.3)**: Good load balancing without overriding signal quality

**Mathematical Effect**: 
- Empty RSU (load=0): load_factor ≈ 166,667, contributes ~50,000 to score
- Half-full RSU (load=5/10): load_factor = 2, contributes 0.6 to score  
- Nearly full RSU (load=9/10): load_factor ≈ 1.1, contributes 0.33 to score

---

### **α₃ = 0.2 (Time to Stay Weight)**

**Code Location**: `vanet_env/environment.py`, line 59
```python
self.alpha3 = 0.2  # Time to stay weight
```

**Purpose**: Encourages connections that will last longer (reduces handover frequency)

**Formula Applied To**: See "Time to Stay Calculation" section below

**Real-world Impact**:
- **Higher values** (0.3-0.5): Fewer handovers, more stable connections
- **Lower values** (0.1): More reactive to immediate signal/load conditions
- **Our choice (0.2)**: Moderate stability preference

**Mathematical Effect**: Every 10 seconds of predicted stay time contributes 2 points to score

---

### **α₄ = 0.1 (Penalty Weight)**

**Code Location**: `vanet_env/environment.py`, line 60
```python
self.alpha4 = 0.1  # Penalty weight
```

**Purpose**: Controls how much bad conditions reduce the handover score

**Real-world Impact**:
- **Higher values** (0.2-0.3): Very conservative, strongly avoids problematic handovers
- **Lower values** (0.05): More tolerant of poor conditions
- **Our choice (0.1)**: Moderate penalty for realistic decision making

**Mathematical Effect**: Every 10 points of penalty reduces final score by 1 point

---

## 📊 **Core Components Calculation**

### **1. Signal Strength Component**

**Base Formula**:
```python
if distance <= 250:  # Coverage radius
    base_signal = 100 * (1 - distance/250)
else:
    base_signal = 0
```

**Code Location**: `vanet_env/environment.py`, lines 239-242

**Negative Effects Applied**:
```python
final_signal = base_signal × fading_effect × weather_effect × shadowing_effect × (1 - interference_penalty) × channel_quality
```

**Real-world Meaning**: 
- **Distance-based**: Signal decreases linearly with distance (realistic for line-of-sight)
- **250m coverage**: Typical RSU coverage radius for VANET applications
- **Degradation factors**: Multiple realistic impairments applied

**Example**:
- At 0m: base_signal = 100
- At 125m: base_signal = 50  
- At 250m: base_signal = 0

---

### **2. Load Factor Component**

**Formula**:
```python
load_factor = 1 / (rsu['current_load'] / rsu['capacity'] + 1e-6)
```

**Code Location**: `vanet_env/environment.py`, line 314

**Real-world Meaning**: 
- **Inverse relationship**: Higher load = lower attractiveness
- **Prevents division by zero**: Small epsilon (1e-6) added
- **Exponential preference**: Empty RSUs much more attractive than full ones

**Load Factor Examples**:
| RSU Load | Load Ratio | Load Factor | Contribution (α₂=0.3) |
|----------|------------|-------------|----------------------|
| 0/10     | 0.0        | ∞ (capped)  | High priority        |
| 1/10     | 0.1        | 10.0        | 3.0                  |
| 5/10     | 0.5        | 2.0         | 0.6                  |
| 8/10     | 0.8        | 1.25        | 0.375                |
| 9/10     | 0.9        | 1.11        | 0.333                |

---

### **3. Time to Stay Component**

**Formula**:
```python
if velocity_towards_rsu > 0:
    predicted_time = (coverage_radius - distance) / velocity_towards_rsu
    # Apply prediction error
    time_to_stay = predicted_time × (1 ± prediction_error)
else:
    time_to_stay = 0  # Moving away from RSU
```

**Code Location**: `vanet_env/environment.py`, lines 287-302

**Real-world Meaning**:
- **Physics-based**: Time = Distance / Velocity
- **Direction-aware**: Only positive if moving towards RSU
- **Prediction uncertainty**: ±20% error simulates real-world unpredictability

**Calculation Steps**:
1. **Direction vector**: `direction_to_rsu = (rsu_pos - vehicle_pos) / distance`
2. **Velocity component**: `velocity_towards_rsu = dot(vehicle_velocity, direction_to_rsu)`
3. **Time prediction**: `time = (250 - current_distance) / velocity_towards_rsu`
4. **Error application**: `final_time = time × random_factor(0.8 to 1.2)`

---

## 🚧 **Detailed Negative Effects**

### **Signal Quality Degradation**

#### **1. Signal Interference (15% per nearby vehicle)**

**Code Location**: `vanet_env/environment.py`, lines 264-272
```python
interference_vehicles = 0
for other_vehicle in self.vehicles:
    if (other_vehicle != {'position': vehicle_pos} and 
        np.linalg.norm(other_vehicle['position'] - rsu_pos) < 100):
        interference_vehicles += 1

interference_vehicles = min(interference_vehicles, self.max_interference_vehicles)
interference_penalty = interference_vehicles * self.interference_factor
final_signal *= (1 - interference_penalty)
```

**Parameters**:
- `self.interference_factor = 0.15` (15% degradation per vehicle)
- `self.max_interference_vehicles = 3` (maximum 3 interfering vehicles)

**Real-world Cause**: Radio frequency interference from other vehicle transmitters

**Impact Examples**:
- 1 nearby vehicle: 15% signal loss
- 2 nearby vehicles: 30% signal loss  
- 3+ nearby vehicles: 45% signal loss (capped)

---

#### **2. Signal Fading (±10% random variation)**

**Code Location**: `vanet_env/environment.py`, lines 246-248
```python
fading_effect = 1 + np.random.normal(0, self.fading_variance)
final_signal *= max(0.1, fading_effect)  # Minimum 10% signal
```

**Parameters**:
- `self.fading_variance = 0.1` (±10% standard deviation)

**Real-world Cause**: Atmospheric conditions, multipath propagation

**Statistical Effect**: 
- 68% of measurements within ±10% of base signal
- 95% of measurements within ±20% of base signal
- Minimum 10% signal maintained to prevent complete loss

---

#### **3. Building Shadowing (up to 10% loss)**

**Code Location**: `vanet_env/environment.py`, lines 251-254
```python
shadowing_effect = 1 - self.building_shadowing * np.random.random()
final_signal *= shadowing_effect
```

**Parameters**:
- `self.building_shadowing = 0.1` (up to 10% loss)

**Real-world Cause**: Buildings and obstacles blocking line-of-sight

**Effect**: Random signal loss between 0-10% depending on urban environment

---

#### **4. Weather Impact (0-30% configurable loss)**

**Code Location**: `vanet_env/environment.py`, lines 249-251
```python
weather_degradation = 1 - self.weather_impact
final_signal *= weather_degradation
```

**Parameters**:
- `self.weather_impact = 0.0` (default: no weather)
- Configurable up to 0.3 (30% loss in severe weather)

**Real-world Cause**: Rain, snow, fog affecting radio propagation

**Weather Examples**:
- Clear weather: 0% impact
- Light rain: 5-10% impact
- Heavy rain: 15-20% impact
- Snow/fog: 20-30% impact

---

### **Network Congestion Effects**

#### **5. RSU Congestion Penalty**

**Code Location**: `vanet_env/environment.py`, lines 322-326
```python
rsu_load_ratio = rsu['current_load'] / rsu['capacity']
if rsu_load_ratio > self.congestion_threshold:
    congestion_penalty = (rsu_load_ratio - self.congestion_threshold) * self.congestion_delay_penalty
```

**Parameters**:
- `self.congestion_threshold = 0.7` (70% load threshold)
- `self.congestion_delay_penalty = 0.3` (30% penalty multiplier)

**Real-world Cause**: Network equipment performance degradation under heavy load

**Penalty Examples**:
| RSU Load | Load Ratio | Above Threshold | Penalty |
|----------|------------|-----------------|---------|
| 7/10     | 0.7        | 0.0             | 0.0     |
| 8/10     | 0.8        | 0.1             | 0.03    |
| 9/10     | 0.9        | 0.2             | 0.06    |
| 10/10    | 1.0        | 0.3             | 0.09    |

---

#### **6. RSU Processing Overload**

**Code Location**: `vanet_env/environment.py`, lines 328-332
```python
if rsu['current_load'] > self.processing_overload_threshold:
    excess_load = rsu['current_load'] - self.processing_overload_threshold
    overload_penalty = excess_load * self.overload_penalty
```

**Parameters**:
- `self.processing_overload_threshold = 8` (8 vehicles threshold)
- `self.overload_penalty = 0.25` (25% penalty per excess vehicle)

**Real-world Cause**: CPU/memory limitations in RSU hardware

**Penalty Examples**:
- 8 vehicles: No penalty
- 9 vehicles: 0.25 penalty
- 10 vehicles: 0.50 penalty
- 12 vehicles: 1.00 penalty

---

#### **7. Processing Delay Penalty**

**Code Location**: `vanet_env/environment.py`, line 335
```python
processing_delay_penalty = self.handover_processing_delay * 0.1
```

**Parameters**:
- `self.handover_processing_delay = 2.0` (2 seconds processing time)

**Real-world Cause**: Authentication, setup, and protocol overhead

**Fixed Impact**: 0.2 penalty for any handover attempt (represents resource cost)

---

### **Handover Complications**

#### **8. Handover Failure (5% random failure)**

**Code Location**: `vanet_env/environment.py`, lines 430-434
```python
handover_fails = np.random.random() < self.handover_failure_rate
if handover_fails:
    reward = -2  # Penalty for failed handover
    self.total_handover_failures += 1
```

**Parameters**:
- `self.handover_failure_rate = 0.05` (5% failure probability)

**Real-world Cause**: Authentication failures, timeouts, network errors

**Impact**: Complete handover failure with -2 reward penalty

---

#### **9. Handover Cooldown Penalty**

**Code Location**: `vanet_env/environment.py`, lines 337-342
```python
if vehicle_id in self.recent_handovers:
    time_since_last = self.current_step - self.recent_handovers[vehicle_id]
    if time_since_last < self.handover_cooldown:
        recent_handover_penalty = 0.3 * (1 - time_since_last / self.handover_cooldown)
```

**Parameters**:
- `self.handover_cooldown = 5.0` (5 seconds minimum interval)

**Real-world Cause**: Network protocols preventing rapid successive handovers

**Penalty Decay**:
- Immediately after handover: 0.3 penalty
- 2.5 seconds later: 0.15 penalty
- 5+ seconds later: 0.0 penalty

---

#### **10. Channel Quality Variations**

**Code Location**: `vanet_env/environment.py`, lines 276-277
```python
final_signal *= self.channel_quality_factor
```

**Parameters**:
- `self.channel_quality_factor = 1.0` (default: perfect quality)
- Configurable from 0.5 to 1.0 (50-100% quality)

**Real-world Cause**: Environmental interference, traffic congestion

**Impact**: Overall signal scaling factor

---

### **Mobility Prediction Errors**

#### **11. Time-to-Stay Prediction Error (±20%)**

**Code Location**: `vanet_env/environment.py`, lines 298-305
```python
if predicted_time > 0:
    error_factor = 1 + np.random.uniform(-self.prediction_error, self.prediction_error)
    predicted_time *= max(0.1, error_factor)
    
    if abs(error_factor - 1) > 0.1:  # Significant error
        self.total_prediction_errors += 1
```

**Parameters**:
- `self.prediction_error = 0.2` (±20% error range)

**Real-world Cause**: Unpredictable vehicle behavior, traffic changes

**Error Examples**:
- Predicted 10 seconds → Actual 8-12 seconds
- Predicted 20 seconds → Actual 16-24 seconds
- Tracks significant errors (>10% deviation) for statistics

---

## 🧮 **Complete Calculation Example**

### **Scenario**: Vehicle approaching RSU with realistic challenges

**Initial Conditions**:
- Distance to RSU: 100m
- RSU load: 8/10 vehicles (80%)
- Vehicle speed: 15 m/s towards RSU
- 2 nearby vehicles causing interference
- Light weather impact (10%)

### **Step-by-Step Calculation**:

#### **1. Base Signal Strength**:
```python
distance = 100m
base_signal = 100 * (1 - 100/250) = 100 * 0.6 = 60
```

#### **2. Apply Negative Effects to Signal**:
```python
# Fading (random ±10%)
fading_effect = 0.95  # Unlucky 5% reduction

# Weather impact (10%)
weather_effect = 1 - 0.1 = 0.9

# Building shadowing (random 0-10%)
shadowing_effect = 1 - 0.1 * 0.7 = 0.93  # 7% loss

# Interference (2 vehicles × 15%)
interference_penalty = 2 * 0.15 = 0.3
interference_effect = 1 - 0.3 = 0.7

# Channel quality (perfect)
channel_quality = 1.0

# Combined signal
final_signal = 60 * 0.95 * 0.9 * 0.93 * 0.7 * 1.0 = 33.2
```

#### **3. Load Factor**:
```python
load_factor = 1 / (8/10 + 1e-6) = 1 / 0.8 = 1.25
```

#### **4. Time to Stay**:
```python
# Base prediction
remaining_distance = 250 - 100 = 150m
predicted_time = 150 / 15 = 10 seconds

# Apply ±20% error (unlucky +15%)
time_with_error = 10 * 1.15 = 11.5 seconds
```

#### **5. Penalties**:
```python
# Base penalty (vehicle already connected)
base_penalty = 0.5

# Congestion penalty (load 0.8 > threshold 0.7)
congestion_penalty = (0.8 - 0.7) * 0.3 = 0.03

# Overload penalty (8 vehicles = threshold, no penalty)
overload_penalty = 0

# Processing delay
processing_penalty = 2.0 * 0.1 = 0.2

# Recent handover penalty (assume none)
recent_handover_penalty = 0

# Failure risk penalty
failure_risk_penalty = 0.05 * 10 = 0.5

# Channel penalty (no degradation)
channel_penalty = 0

# Total penalty
total_penalty = 0.5 + 0.03 + 0 + 0.2 + 0 + 0.5 + 0 = 1.23
```

#### **6. Final Score**:
```python
score = 0.4 * 33.2 + 0.3 * 1.25 + 0.2 * 11.5 - 0.1 * 1.23
      = 13.28 + 0.375 + 2.3 - 0.123
      = 15.83
```

#### **7. Additional Penalties**:
```python
# Poor signal penalty (signal < 10? No, signal = 33.2)
# Near capacity penalty (load > 0.9? No, load = 0.8)
final_score = 15.83
```

### **Comparison with Ideal Conditions**:

**Without negative effects**:
```python
ideal_signal = 60  # No degradation
ideal_time = 10    # No prediction error
ideal_penalty = 0.5  # Only base penalty

ideal_score = 0.4 * 60 + 0.3 * 1.25 + 0.2 * 10 - 0.1 * 0.5
            = 24 + 0.375 + 2 - 0.05
            = 26.325
```

**Impact of negative effects**: 
- Realistic score: 15.83
- Ideal score: 26.325  
- **Degradation: 39.9%** - Nearly 40% reduction due to realistic challenges!

This demonstrates how significantly the negative parameters affect handover decisions, making the RL agent's job much more challenging and realistic.

---

## 🎛️ **Parameter Tuning Guidelines**

### **For Different Applications**:

#### **High-Speed Highway**:
- ↑ α₁ = 0.6 (prioritize signal strength)
- ↓ α₃ = 0.1 (less stability needed)
- ↓ congestion_threshold = 0.5 (avoid congestion)

#### **Dense Urban Traffic**:
- ↑ α₂ = 0.5 (strong load balancing)
- ↑ interference_factor = 0.25 (more interference)
- ↑ building_shadowing = 0.2 (more obstacles)

#### **Critical Safety Applications**:
- ↑ α₄ = 0.3 (very conservative)
- ↓ handover_failure_rate = 0.01 (less tolerance for failure)
- ↓ congestion_threshold = 0.6 (avoid overload)

#### **Best Performance Testing**:
- Set all negative effects to minimum values
- Focus on core algorithm performance

This comprehensive guide shows how every parameter contributes to realistic and challenging handover decisions that prepare the RL agent for real-world VANET deployments. 