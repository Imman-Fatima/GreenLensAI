# GreenLensAI – AI-Based Plant Classification App

## Project Overview
GreenLensAI is an AI-based Android application that identifies flowers and indoor plants using Machine Learning and TensorFlow Lite.

## Objectives
- Identify flowers and indoor plants using AI.
- Perform on-device image classification.
- Provide plant information and care tips.
- Allow users to capture images or select them from the gallery.

## Features
- Scan a plant using the camera.
- Select an image from the gallery.
- Display predicted plant name.
- Show confidence score, scientific name, category, description, and care tips.
- Work with an integrated TensorFlow Lite model.

## Technologies Used
- Java
- XML
- Android Studio
- TensorFlow
- TensorFlow Lite
- MobileNetV2
- Google Colab
- Python

## AI Model
- Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224
- Number of Classes: 15
- Final Test Accuracy: 89.79%
- Model Format: TensorFlow Lite (.tflite)

## Supported Classes
AloeVera, Aralia, Calathea, KaminiBonsai, LuckyBamboo, MoneyPlant, Monstera, PurpleHeart, SnakePlant, SpiderPlant, Daisy, Dandelion, Roses, Sunflowers, Tulips.

## How to Run
1. Clone or download this repository.
2. Open the project in Android Studio.
3. Allow Gradle synchronization to complete.
4. Connect an Android phone or start an emulator.
5. Run the application.

## Original Reference Repository
https://github.com/Uzair390-Del/NumberGuessingAppTflite

This repository was studied as a reference for understanding TensorFlow Lite integration in Android applications.

## Project Name
GreenLensAI – Discover Nature with AI
