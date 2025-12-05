# CQ_CYBERBROS
Repository for CodeQuest Hackathon solution

FISIERE NECESARE:
-  yolov12m-face.pt
-  yolo12s.pt
-  fsrcnn_x3.pth
-  model_mobilefacenet.pth

Fisierele se vor rula in felul urmator:
- avem nevoie de Yolo.py ( aici au fost scrise functiile de run inference train - unde folosim yolo pe fata ) si test unde avem si pe persoane, si pe fata
- crate_db_mobilefacenet.py - pentru formarea bazei de date
- full_pipeline_test_FSR_mobilefacenet_time - pt pipeline
- inference_cristi -> am pus poza cu colegul nostru ( fara accesorii ) in baza de date , inferenta pe o alta poza cu colegul nostru cu accesorii.
