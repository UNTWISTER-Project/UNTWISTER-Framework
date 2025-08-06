<!-- Banner -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/8535c2bd-b158-4058-8d52-b16fe44d386d" alt="UNTWISTER Banner" width="800">
</p>

# UsiNg digital TWIns to enable SecuriTy in cybER-physical ecosystems (UNTWISTER)
### platform_manager.py

The **Platform Manager** is responsible for orchestrating the initialization of the framework. It loads configuration parameters and adversary models from JSON files, distributes these to the other modules via REST calls, optionally receives a trained AI model from the `model_creation` module, and finally triggers the start of the working phase.

Main capabilities:
- Loads configuration files (`configuration_parameters.json`, `adversary_model.json`) and merges them into a single setup dictionary.
- Sends the appropriate JSON configuration to each component (Data Collection, Detection & Classification, Attack Mitigation, Reconfiguration).
- Receives and forwards the AI model file (e.g., `.pth` weight file) to the Detection & Classification module.
- Issues a **START_WORKING_PHASE** command to the Data Subscription module to begin the real-time interaction loop.

> Endpoint: `/start_platform_from_files`  
> Default port: `5000`
>
> ### data_subscription_and_collection.py

The **Data Subscription & Collection** module is responsible for receiving runtime data from both the Physical System and the Digital Twin via RabbitMQ queues, synchronizing messages by simulation step, and forwarding aligned input batches to the Detection & Classification component.

Main capabilities:
- Receives configuration of sensors/queues through a dedicated REST endpoint.
- Subscribes to multiple RabbitMQ queues in parallel (one per sensor/device), consuming messages asynchronously.
- Keeps a per‐timestep buffer and only forwards data once corresponding measurements from all configured queues are available.
- Automatically aligns Digital Twin and Physical System data by `simstep`, merges them into a single JSON structure, and POSTs it to the Detection & Classification module.
- Optionally logs all incoming data to CSV files for offline analysis and plotting.

> Configuration endpoint: `/data_collection`  
> Start endpoint: `/start_work`  
> Forwarding target: `http://localhost:5002/start_work`  
> Default port: `5001`

### detection_and_classification.py

This module hosts the AI-based **anomaly detection and attack classification engine** of the framework. It loads a pre‐trained MLP model and performs real‐time inference on incoming, time‐aligned data from the physical system and the Digital Twin.

Main capabilities:
- Accepts a sliding window of synchronized sensor readings and DT predictions.
- Extracts statistical features (mean, variance) over each window.
- Executes a PyTorch MLP (classification + regression head) to:
  - **Detect anomalies**
  - **Classify the type of attack**
  - **Estimate the intensity** of the ongoing manipulation (regression head smoothed using an EMA).
- Maintains a logic that converts transient detections into *true attacks* by tracking a configurable number of consecutive positive windows.
- Once an attack is confirmed, it:
  - Sends a *STOP signal* to the physical system via RabbitMQ.
  - Computes a set of “state variables”.
  - Sends the attack classification (class + regression+state) to the `attack_mitigation` component.

Supporting endpoints:
- `/detection_classification` – configuration
- `/detection_classification/upload` – upload of the `.pth` model file
- `/start_work` – real‐time inference loop (triggered by Data Subscription)
  
> Default port: `5002`

### attack_mitigation.py

This component receives the detected attack information (class, estimated parameter, and system state) from the Detection & Classification module and performs **predictive simulations** to assess whether the ongoing attack may evolve into a **critical scenario**.

Main capabilities:
- Receives anomaly characterization (attack class + regression value + state variables) via the `/mitigate` endpoint.
- Dynamically updates a *co-simulation configuration JSON* (`mm.json`) with the current state and estimated attack parameter.
- Launches a short **INTO-CPS Maestro2 predictive co-simulation**, starting from the current system state.
- Parses the predicted evolution and checks whether **critical conditions** are met (e.g., time‐to‐collision).
- If a critical event is predicted, it sends a command to the **Reconfiguration** component to trigger an appropriate counter‐measure.

Supporting endpoints:
- `/attack_mitigation` – configuration
- `/mitigate` – receives attack data and executes prediction
- `/start_work` – placeholder start endpoint

> Default port: `5003`

### reconfiguration.py

The **Reconfiguration** component is responsible for enforcing corrective actions on the physical system once a critical scenario has been predicted by the mitigation layer.

Main capabilities:
- Receives mitigation commands (acceleration/deceleration or stop signals) via the `/start_work` endpoint.
- Retrieves the most recent **real-time state** of the system (via the Data Collection module) to compute safe actuation profiles.
- Evaluates whether the corrective action (e.g., emergency braking) can be safely applied without inducing new unsafe conditions.
- Constructs a control message with updated actuation commands and publishes it towards the physical system via RabbitMQ.
- Supports configurable mapping of **exchange / queue / routing_key** per physical sensor/actuator.

Supporting endpoints:
- `/reconfiguration` – configuration
- `/start_work` – receives mitigation command and sends the actuation control message

> Default port: `5005`

## Quick Start

> The UNTWISTER framework is composed of six Flask-based microservices.  
> Each service runs independently and communicates via RabbitMQ and REST APIs.

### Prerequisites

- Python ≥ 3.8  
- RabbitMQ running locally (`localhost:5672`)  
- INTO-CPS Maestro2 (for simulation-based validation)  

Install Python requirements:

```bash
pip install -r requirements.txt
```

### Step 1 — Start RabbitMQ

```bash
sudo service rabbitmq-server start
```

### Step 2 — Launch UNTWISTER services

_Open six terminal windows (or use tmux) and run:_

```bash
# 1) Platform Manager
python platform_manager.py

# 2) Data Subscription & Collection
python data_subscription_and_collection.py

# 3) Detection & Classification
python detection_and_classification.py

# 4) Attack Mitigation
python attack_mitigation.py

# 5) Reconfiguration
python reconfiguration.py

# (Optional) 6) Model Creation / Training Module
python model_creation/model_creation.py
```

### Step 3 — Configure and start the working phase

```bash
curl -X POST http://localhost:5000/start_platform_from_files
```

This instructs the platform manager to:
- load `configuration_parameters.json` and `adversary_model.json`
- send configuration to all services
- broadcast the `START_WORKING_PHASE` command
