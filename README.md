AI Interview Stress Analyzer
Overview

The AI Interview Stress Analyzer is an intelligent multimodal system designed to evaluate a candidate’s psychological state during interview scenarios.
The platform analyzes facial expressions, voice patterns, and speech signals to estimate stress levels using deep learning and real-time processing.

This system combines computer vision, audio signal processing, and machine learning to provide measurable, data-driven feedback instead of subjective human evaluation.

Problem Statement

Traditional interview assessments rely heavily on subjective human judgment. Psychological indicators such as:

Facial micro-expressions

Voice tremors

Speech rhythm variation

Emotional instability

are often overlooked or inconsistently evaluated.

This system aims to:

Objectively quantify stress levels

Provide real-time feedback

Support structured performance analysis

System Architecture

The system follows a modular AI pipeline architecture:

Client (Web Interface)
        ↓
FastAPI Backend
        ↓
Multimodal Processing Layer
        ↓
CNN + LSTM Deep Learning Models
        ↓
Stress Classification Engine
        ↓
Report Generation (PDF)

Core Modules

Video Processing Module

Face detection using MediaPipe

Frame extraction via OpenCV

Feature encoding using CNN

Audio Processing Module

Audio extraction

MFCC feature generation (Librosa)

Temporal modeling using LSTM

Stress Classification Model

CNN for spatial features

LSTM for temporal dependencies

Fully connected layer for stress prediction

Backend API

RESTful endpoints (FastAPI)

File upload handling

Real-time inference

Report Generation

Structured PDF generation using ReportLab

Stress score summary

Performance insights

Technologies Used
Backend

Python 3.10

FastAPI

Uvicorn

Pydantic

AI / ML

PyTorch

TorchVision

Torchaudio

Scikit-learn

CNN + LSTM architecture

Computer Vision

OpenCV

MediaPipe

Audio Processing

Librosa

SoundFile

Data Analysis & Visualization

Pandas

Matplotlib

Seaborn

Report Generation

ReportLab

Deep Learning Model Flow
CNN Component

Extracts spatial facial features from video frames

Captures micro-expressions and muscle tension patterns

LSTM Component

Processes sequential temporal data

Identifies stress-related fluctuations over time

Final Classification

Outputs stress probability score

Categorizes stress levels:

Low

Moderate

High

API Design
Base URL
http://localhost:8000

Endpoints
Upload Interview Recording
POST /analyze


Input: Video or audio file
Output: Stress analysis JSON response

Generate Report
POST /generate-report


Output: Downloadable PDF file

Installation
1. Clone Repository
git clone <your-repo-url>
cd Interview-Stress-Analyzer

2. Create Virtual Environment
python -m venv ai_env
ai_env\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run Server
uvicorn main:app --reload

Engineering Design Principles

Modular architecture for scalability

Separation of AI inference and API logic

Reproducible environment via version locking

Real-time inference capability

Extensible for additional behavioral signals

Future Improvements

Transformer-based temporal modeling

Real-time WebSocket stress monitoring

Interview feedback recommendation engine

Cloud deployment (AWS / GCP)

Model quantization for edge deployment

Use Cases

Interview preparation platforms

HR evaluation tools

Behavioral research studies

Communication skill training systems

License

This project is intended for academic and research purposes.