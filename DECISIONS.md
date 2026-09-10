# Decisions Log

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Using uv for environment management | Available on system, fast, handles Python version management | Build start |
| 2 | YOLOv8n pretrained (COCO) for vehicle detection | COCO already contains car/bus/truck/motorcycle classes; plate detection via dedicated model or vehicle crop fallback | Build start |
| 3 | SQLite for database | Simple, no server needed, perfect for demo | Build start |
| 4 | **EasyOCR** instead of PaddleOCR | PaddleOCR 2.10 has torch DLL conflicts on Windows (`shm.dll` WinError 127). EasyOCR installed cleanly and works with torch 2.5.1+cpu. Trade-off: slightly lower OCR accuracy on synthetic plates, but functional end-to-end. | Build - PaddleOCR failed |
| 5 | ByteTrack via deep-sort-realtime for per-camera tracking | Lightweight, well-tested, works in real-time | Build start |
| 6 | RapidFuzz for fuzzy plate matching | Fast C-based Levenshtein, better than python-Levenshtein for our needs | Build start |
| 7 | Pretrained yolov8n.pt for vehicle detection only | No fine-tuned plate model available yet; plate detection disabled at detection stage, downstream OCR attempts on vehicle crops instead | Build start |
| 8 | Synthetic test videos | No real traffic footage available; created programmatically generated videos with labeled plates for end-to-end pipeline testing. Real footage needed for demo. | Build start |
| 9 | torch 2.5.1+cpu pinned | torch 2.14.0 caused WinError 127 on shm.dll. Downgrading to 2.5.1+cpu resolved the issue. | Build - torch fix |
| 10 | start_demo.py (Python) as primary launcher | Cross-platform; start_demo.sh also provided for Linux/macOS/WSL | Build start |
