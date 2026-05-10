# Practical-Data-science-GR5

#  Analyzing Patient Records — RAMA-AS Improvement Project

> Repository นี้เป็นส่วนหนึ่งของรายวิชา **ITDS346 Practical Data Science**  
> โครงงานนี้มุ่งเน้นการวิเคราะห์ข้อมูลผู้ป่วยเพื่อปรับปรุงระบบช่วยประเมินความเสี่ยงโรคไส้ติ่งอักเสบ โดยอ้างอิงแนวคิดจาก **RAMA-AS (Ramathibodi Appendicitis Score)**

---

##  Project Overview

โครงงานนี้ศึกษาและพัฒนาแนวทางการวิเคราะห์ข้อมูลผู้ป่วยที่สงสัยภาวะ **Appendicitis** หรือ **ไส้ติ่งอักเสบ** โดยใช้ข้อมูลอาการทางคลินิกและผลตรวจทางห้องปฏิบัติการ เพื่อช่วยสนับสนุนการตัดสินใจเบื้องต้นของแพทย์

แนวทางหลักของโครงงานคือการนำสมการ RAMA-AS เดิมมาตรวจสอบกับ dataset ปัจจุบัน และทดลองพัฒนาโมเดลใหม่ด้วยเทคนิคด้าน Data Science เช่น

- Logistic Regression
- SHAP Explainability
- PCA / MCA
- Q-Learning
- Deep Q-Network (DQN)

---

##  Objectives

- ศึกษากระบวนการทำงานด้าน Data Science ตั้งแต่การเตรียมข้อมูลจนถึงการประเมินผลโมเดล
- วิเคราะห์ความเหมาะสมของ feature และ weight เดิมจาก RAMA-AS
- เปรียบเทียบโมเดลหลายแนวทางเพื่อหาวิธีที่เหมาะสมกับข้อมูลจริง
- อธิบายบทบาทของแต่ละ feature ด้วย SHAP และ Feature Importance
- ทดลองใช้ Reinforcement Learning เพื่อศึกษาพฤติกรรมการตัดสินใจของโมเดล

---

##  Dataset & Features

ข้อมูลที่ใช้เป็นข้อมูลผู้ป่วยที่ผ่านการ **Anonymization** แล้ว โดยนำมาแปลงให้อยู่ในรูปแบบที่เหมาะสมต่อการวิเคราะห์เชิง Data Science

### Main Features

| Feature | Description |
|---|---|
| Progression | อาการปวดมีความรุนแรงหรือดำเนินเพิ่มขึ้น |
| Migration | อาการปวดย้ายตำแหน่ง |
| Aggravation | อาการปวดรุนแรงขึ้นเมื่อเคลื่อนไหว |
| TEMP | อุณหภูมิร่างกาย / ไข้ |
| Rebound | อาการกดปล่อยเจ็บ |
| WBC | ค่าเม็ดเลือดขาว |
| Neut | ค่า Neutrophil |
| Appendicitis | Target variable |

ข้อมูลส่วนใหญ่ถูกแปลงเป็นรูปแบบ **Binary Feature (0/1)** เพื่อให้เหมาะกับ Logistic Regression, MCA และ Reinforcement Learning

---

##  Data Preparation

ขั้นตอนการเตรียมข้อมูลประกอบด้วย

1. ตรวจสอบ metadata ของข้อมูล  
2. ตรวจสอบ missing values และค่าผิดปกติ  
3. แปลงข้อมูลอาการให้เป็นค่า 0/1  
4. ตรวจสอบ outliers  
5. เตรียม feature set สำหรับทดลองโมเดล  
6. เปรียบเทียบผลระหว่างกรณีใช้ feature ครบ และกรณีตัดบาง feature ออก  

---

##  Feature Selection & SHAP Analysis

จากการวิเคราะห์ด้วย Logistic Regression และ SHAP พบว่า feature แต่ละตัวมีบทบาทต่อโมเดลไม่เท่ากัน

### Key Findings

- **Rebound, WBC และ Neut** เป็นกลุ่ม feature ที่มีอิทธิพลสูงต่อการทำนาย
- **TEMP** มี contribution ต่ำกว่าที่คาดไว้เมื่อเทียบกับ feature อื่น
- เมื่อทดลองตัด TEMP ออก พบว่าโมเดลบางกรณีมี AUC ลดลง แสดงว่า TEMP ยังมี contribution อยู่
- Migration และ Aggravation มีผลต่อโมเดลค่อนข้างต่ำเมื่อเทียบกับ feature หลัก

สรุปคือ TEMP อาจไม่ใช่ feature ที่สำคัญที่สุด แต่ยังช่วยรักษาประสิทธิภาพของโมเดลโดยรวม

---

##  Models Used

### 1. RAMA-AS Baseline

ใช้สมการ RAMA-AS เดิมเป็น baseline สำหรับเปรียบเทียบกับโมเดลใหม่

---

### 2. Logistic Regression

ใช้สำหรับ train โมเดลใหม่จาก dataset ปัจจุบัน เพื่อดูว่า weight ของ feature เปลี่ยนไปจากสมการเดิมหรือไม่

---

### 3. Adaptive RAMA-AS with MCA

เนื่องจากข้อมูลเป็น Binary Feature จึงใช้ **Multiple Correspondence Analysis (MCA)** แทน PCA เพื่อวิเคราะห์ pattern ของข้อมูล categorical/binary

แนวทาง MCA ถูกใช้ใน 3 ส่วนหลัก

| Method | Purpose |
|---|---|
| Adaptive Feature Weighting | ปรับ weight ของ RAMA-AS ให้เหมาะกับ dataset |
| Sequential Pattern Analysis | ตรวจ Concept Drift |
| Outlier Detection | ตรวจผู้ป่วยที่มี pattern ผิดปกติ |

---

### 4. Q-Learning

ใช้แนวคิด Reinforcement Learning แบบ Q-Table โดยแปลง feature 7 ตัวเป็น state ทั้งหมด `2^7 = 128 states`

ตัวอย่าง state encoding:

```python
[Progression=1, Migration=0, Aggravation=1, TEMP=1, Rebound=0, WBC=1, Neut=1]
=> "1011011"
=> state = 91


---

### 5. Deep Q-Network (DQN)

ใช้ Neural Network แทน Q-Table เพื่อให้โมเดลสามารถ generalize ได้ดีขึ้น

Network structure:

```text
Input Layer: 7 features
Hidden Layer 1: 128 neurons + ReLU + Dropout
Hidden Layer 2: 64 neurons + ReLU + Dropout
Output Layer: 2 actions
```

---

##  Final Model Comparison

ผลการทดลองบนข้อมูลจริงจำนวน **376 records**

| Metric      | RAMA-AS Baseline | Q-Learning |    DQN | Adaptive RAMA-AS MCA | Best         |
| ----------- | ---------------: | ---------: | -----: | -------------------: | ------------ |
| Accuracy    |           76.33% |     77.13% | 78.46% |               76.33% | DQN          |
| Sensitivity |           90.23% |     87.22% | 89.10% |               93.98% | Adaptive MCA |
| Specificity |           42.73% |     52.73% | 52.73% |               33.64% | QL / DQN     |
| F1-Score    |           0.8436 |     0.8436 | 0.8541 |               0.8489 | DQN          |
| AUC         |           0.7555 |     0.6796 | 0.7236 |               0.7587 | Adaptive MCA |
| FP          |               63 |         52 |     52 |                   73 | QL / DQN     |
| FN          |               26 |         34 |     29 |                   16 | Adaptive MCA |

---

##  Key Results

### Adaptive RAMA-AS MCA

จุดเด่นคือสามารถลด **False Negative (FN)** ได้ดีที่สุด

* FN ลดเหลือ 16 ราย
* Sensitivity สูงสุดที่ 93.98%
* AUC สูงสุดที่ 0.7587

เหมาะกับบริบททางคลินิกที่ต้องการลดโอกาสพลาดผู้ป่วยจริง

---

### DQN

DQN ให้ผลโดยรวมดีที่สุดในแง่ balance ระหว่าง metric หลายตัว

* Accuracy สูงสุด 78.46%
* F1-Score สูงสุด 0.8541
* Specificity สูงกว่า baseline
* FP ลดจาก 63 เหลือ 52 ราย

เหมาะกับกรณีที่ต้องการลดการวินิจฉัยเกิน และยังรักษาความแม่นยำโดยรวมได้ดี

---

### Q-Learning

Q-Learning มีข้อดีคือเรียบง่าย อธิบายง่าย และลด FP ได้ดี

* Specificity สูงกว่า baseline
* FP ลดลงจาก 63 เหลือ 52 ราย
* แต่ Sensitivity ต่ำกว่า baseline

---

##  Feature Importance

จาก Permutation Feature Importance ของ DQN พบว่า feature ที่สำคัญที่สุดคือ

| Rank | Feature     | Interpretation                        |
| ---- | ----------- | ------------------------------------- |
| 1    | WBC         | ตัวชี้วัดการอักเสบที่สำคัญที่สุด      |
| 2    | Rebound     | อาการกดปล่อยเจ็บมีผลต่อการวินิจฉัยสูง |
| 3    | Progression | ช่วยบอกลักษณะการดำเนินของอาการ        |
| 4    | Neut        | สะท้อนการติดเชื้อ / การอักเสบ         |
| 5    | TEMP        | มีผลปานกลาง ยังมี contribution        |
| 6    | Migration   | มีผลปานกลาง                           |
| 7    | Aggravation | มีผลน้อยที่สุดในชุดข้อมูลนี้          |

---

##  Repository Structure

```text
Practical-Data-science-GR5/
│
├── DATASET/
│   └── dataset files
│
├── FeatureSelection&Experiment/
│   └── model experiments
│
└── README.md
```

---

##  Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Matplotlib
* MCA / PCA
* Reinforcement Learning
* Deep Q-Network
* Alteryx

---

##  Conclusion

โครงงานนี้แสดงให้เห็นว่า การนำสมการ RAMA-AS เดิมมาใช้กับ dataset ใหม่ควรมีการตรวจสอบและปรับปรุงให้เหมาะสมกับข้อมูลจริง เนื่องจากบทบาทของ feature และ weight อาจเปลี่ยนแปลงตาม distribution ของข้อมูล

ผลการทดลองพบว่า

* **Adaptive RAMA-AS MCA** เหมาะกับการลด False Negative และเพิ่ม Sensitivity
* **DQN** ให้ผลลัพธ์โดยรวมดีและสมดุลที่สุด
* **Q-Learning / DQN** ช่วยเพิ่ม Specificity และลด False Positive ได้ดี
* **WBC และ Rebound** เป็น feature ที่มีผลต่อโมเดลมากที่สุด
* TEMP แม้ไม่ใช่ feature หลักที่สุด แต่ยังมี contribution ต่อประสิทธิภาพของโมเดล

---

##  Contributors

| Student ID | Name                |
| ---------- | ------------------- |
| 6687006    | Kongphop Khaochot   |
| 6687020    | Thanet Somchuewiang |
| 6687067    | Peeranat Machaiwet  |

---

##  References

* Wilasrusmee, C. et al. (2017). Developing and validating of Ramathibodi Appendicitis Score (RAMA-AS) for diagnosis of appendicitis in suspected appendicitis patients.
* Bellman, R. E. (1961). Adaptive Control Processes.
* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.
* Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling.

---

## ⚠️ Disclaimer

This project is for educational purposes in the course **ITDS346 Practical Data Science**.
The model results should not be used as a standalone medical diagnosis tool.


