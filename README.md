# xray-abnormality-detector-with-fastshap
This is my final year project spanning 2 semesters (Semester 2 2024 and Semester 1 2025).

A two-stage abnormality detector in musculoskeletal radiographs using DeiT-B as backbone for both stages, fine-tuned on MURA dataset via transfer learning. Stage 1 classifies radiographs by making predictions about the body parts and channels them to their corresponding binary abnormality detector in stage 2. The model is paired with [fastSHAP](https://github.com/iancovert/fastshap) visualisation for individual explanability behind every normal/abnormal decision.

Examples:
Red regions indicate positive contribution for the predicted labels, whereas blue regions indicate contribution against the predicted labels.
<img width="2700" height="900" alt="shap_XR_SHOULDER_pred (2)" src="https://github.com/user-attachments/assets/a07539f6-0b95-4c41-9ede-93d2453e6a68" />
<img width="2700" height="900" alt="shap_XR_WRIST_pred (2)" src="https://github.com/user-attachments/assets/1870d778-f59f-47ac-9e70-eb3b211994fb" />
<img width="2700" height="900" alt="shap_XR_HUMERUS_pred (2)" src="https://github.com/user-attachments/assets/09a77166-08da-4aa1-a05d-cc0db8e6a153" />

References and credits:

N. Jethani, M. Sudarshan, I. Covert, S.-I. Lee, and R. Ranganath, "FastSHAP: Real-Time Shapley Value
Estimation," presented at the International Conference on Learning Representations 2022, Virtual,
2022.

I. Covert, C. Kim, and S.-I. Lee, "Learning to estimate Shapley values with vision transformers,"
presented at the International Conference on Learning Representations 2023, Kigali, Rwanda, 2023.
