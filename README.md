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
