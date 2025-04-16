
# Uzbek STT + NER Pipeline

**An open-source NLP pipeline for the Uzbek language combining Speech-to-Text (STT) and Named Entity Recognition (NER).**  
Developed as part of an applied machine learning research initiative to support low-resource languages in real-world scenarios.

## 🧠 Overview

This repository presents an end-to-end Uzbek NLP system that transcribes spoken Uzbek into text and extracts meaningful named entities. It is designed with a modular architecture, enabling easy experimentation and deployment in speech-driven applications such as virtual assistants, automatic subtitles, and conversational AI.



---

## 🚀 Features

- 🎙️ **Speech-to-Text (STT):** 
  - Fine-tuned the Whisper-v2-large model for Uzbek
  - Handles dialectal variation and noisy recordings
  - Supports `.wav` input with ~% WER on validation set

- 🔍 **Named Entity Recognition (NER):** 
  - Custom-trained BERT-based model
  - Recognizes `PER` (Person), `LOC` (Location), `ORG` (Organization), `DATE`, and `MISC`
  - Annotated dataset tailored for Uzbek morphology and entity syntax

- ⚙️ **Pipeline Integration:**
  - Accepts speech/audio input
  - Outputs structured entities with confidence scores
  - Ready for REST API deployment or CLI use


---

## 🧪 Example Usage

```bash
# Run the full pipeline
python pipeline.py-- input path/to/audio.wav
```

**Output:**

```json
{
  "transcript": "Shavkat Mirziyoyev Toshkentda chiqish qildi.",
  "entities": [
    {"text": "Shavkat Mirziyoyev", "type": "PER", "start": 0, "end": 20},
    {"text": "Toshkent", "type": "LOC", "start": 21, "end": 29}
  ]
}
```

---

## 📊 Model Performance

| Task      | Model          | Metric     | Value     |
|-----------|----------------|------------|-----------|
| STT       | Whisper-v2-Large      | WER        | ~%      |
| NER       | BERT-multilingual (Uz)| F1-score   | ~%      |

> 🧪 Evaluated on a held-out Uzbek validation set (1000 samples, mixed domain)

---

## 📚 Dataset Sources

- **STT:** Custom-collected Uzbek speech corpus (open-source + supplemented)
- **NER:** Annotated sentences from news, Wikipedia, and public admin texts





## 📜 License

This project is licensed under the MIT License.  
For research use only. Commercial use requires permission.

---


![image](https://github.com/user-attachments/assets/1d61fc27-4829-4f41-a1b9-864361c1518e)

![image](https://github.com/user-attachments/assets/a2bcf58d-0d0b-4b61-a696-60eaaa751379)

![image](https://github.com/user-attachments/assets/29c1ece5-b496-4c66-9077-a70513851227)


